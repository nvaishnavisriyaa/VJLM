"""Phase 8 - LoRA / QLoRA fine-tune, run locally on a CUDA GPU.

The notebooks under training/notebooks/ target Colab: /content/ paths, Drive
checkpoints, and a free T4. This runs the FT-001 configuration on a local
card, so Colab is no longer in the loop (FOUND-003 allowed Colab because no
local CUDA GPU existed; that is no longer true).

Precision is DETECTED, not assumed. HANDOFF.md 4 says "use fp16 on a T4"
because Turing has no native bf16 and emulates it - that is a T4 workaround,
not a general rule. On Ampere or newer, bf16 is correct and more stable.

Per experiments/PLAN.md, Qwen2.5-1.5B is the CONTROL arm, not the selected
model - it exists to prove the training path end to end. The base model is
chosen at Phase 7 from a measured comparison, not from this default.

Usage:
    python training/train_local.py                                  # control
    python training/train_local.py --base <hf-id> --epochs 3        # LoRA
    python training/train_local.py --base <hf-id> --qlora           # QLoRA

Why this script tokenizes the dataset itself
--------------------------------------------
trl is handed a PRE-TOKENIZED dataset (`input_ids` + `completion_mask`), which
makes it skip its own preparation. That is deliberate - both of its own paths
are unsafe for this data:

  * The *standard* prompt-completion path tokenizes with the tokenizer default
    `add_special_tokens=True`. Our prompt string already carries a literal
    <s> from the chat template, so that produces TWO BOS tokens. Measured on
    Mistral-7B-v0.3: 2 BOS with the default, 1 with add_special_tokens=False.
    Qwen2.5 has no BOS at all, which is exactly why the control arm never
    exposed this and why it would have silently degraded every new arm.

  * The *conversational* path templates prompt+completion together to find the
    completion boundary. Mistral-7B-Instruct's template injects the system
    message next to the LAST user turn, so appending an assistant turn moves
    it and the two renders diverge (measured: at character 10). trl only logs
    a warning and then builds a MISALIGNED completion mask from it.

So the pair is constructed instead of scraped:

    prompt     = apply_chat_template([system, user], add_generation_prompt=True)
    completion = assistant_answer + eos

That is byte-identical to what rag/retrieve.py sends at serving time, is
prefix-stable by construction for every model, and teaches the model to stop.
training/check_template.py verifies all of it and must pass before training.
"""
import argparse, glob, hashlib, inspect, json, os, random, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "training", "data")
ADAPTERS = os.path.join(ROOT, "training", "adapters")
sys.path.insert(0, ROOT)

from training.check_template import apply_template, pick_pad  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# FT-001: all attention + MLP projections. Kept as a module constant so the
# A-vs-C comparison cannot drift - both arms must target the same modules, and
# PEFT must never be left to choose implicitly.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def lora_kwargs(r, alpha, dropout, target_modules):
    """FT-001 defaults, exposed so PLAN.md's r=8/16/32 sweep needs no edit."""
    return dict(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=list(target_modules))


def assert_no_leakage():
    """The Phase 8 gate is 'check for train/test leakage'. Cheap to verify, and
    a silent regression here invalidates every number the run produces."""
    def load(name):
        p = os.path.join(DATA, name + ".jsonl")
        return [json.loads(l) for l in open(p, encoding="utf-8")]

    train, val, test = load("train"), load("val"), load("test")
    pair = lambda e: (e["messages"][1]["content"], e["messages"][2]["content"])
    tr = set(map(pair, train))
    bad = sum(1 for e in val + test if pair(e) in tr)
    if bad:
        sys.exit("ABORT: %d val/test examples duplicate a train example. "
                 "Re-run training/build_dataset.py." % bad)
    print("  leakage check: clean (train=%d val=%d test=%d)" % (len(train), len(val), len(test)))


def pick_precision(torch):
    """bf16 where the hardware supports it natively, fp16 otherwise."""
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    bf16 = torch.cuda.is_bf16_supported()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("  GPU: %s (SM %d.%d, %.1f GB)" % (name, cap[0], cap[1], vram))
    print("  bf16 supported: %s -> using %s" % (bf16, "bf16" if bf16 else "fp16"))
    if not bf16:
        print("  (pre-Ampere card: bf16 would be emulated and very slow - see HANDOFF 4)")
    return bf16


def filter_kwargs(cls, want, aliases):
    """transformers/trl rename config kwargs across versions; keep what the
    installed signature actually accepts. Same approach as colab_compact.py."""
    ok = set(inspect.signature(cls.__init__).parameters)
    out = {}
    for k, v in want.items():
        if k in ok:
            out[k] = v
        elif aliases.get(k) in ok:
            out[aliases[k]] = v
        else:
            print("  dropped unsupported arg: %s" % k)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def git_sha():
    """Commit the run was produced at. Provenance, so a recorded number can be
    traced back to the code that made it - so it is worth finding git even when
    it is not on PATH, which it is not inside these venv shells."""
    exes = ["git", r"C:\Program Files\Git\cmd\git.exe"]
    for exe in exes:
        try:
            return subprocess.check_output([exe, "rev-parse", "--short", "HEAD"],
                                           cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
    return None


def pkg_versions():
    out = {}
    for m in ("torch", "transformers", "trl", "peft", "datasets",
              "accelerate", "bitsandbytes"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def build_pairs(tok, split, max_length):
    """Tokenize one split into trl's pre-tokenized form.

    Returns (rows, stats). Each row is {input_ids, completion_mask}; the
    collator turns completion_mask==0 into labels of -100, so the prompt and
    any padding contribute no loss and only the answer is trained on.
    """
    path = os.path.join(DATA, split + ".jsonl")
    rows, truncated, variants, n_bos_bad = [], 0, set(), 0
    for line in open(path, encoding="utf-8"):
        msgs = json.loads(line)["messages"]
        prompt, variant = apply_template(tok, msgs[:-1], add_generation_prompt=True)
        variants.add(variant)
        completion = msgs[-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if tok.bos_token_id is not None:
            if sum(1 for t in p_ids if t == tok.bos_token_id) != 1:
                n_bos_bad += 1
        ids = p_ids + c_ids
        mask = [0] * len(p_ids) + [1] * len(c_ids)
        if len(ids) > max_length:
            truncated += 1
            ids, mask = ids[:max_length], mask[:max_length]
        if 1 not in mask:
            sys.exit("ABORT: an example in %s has no trainable completion token "
                     "after truncation. Raise --max-length." % split)
        rows.append({"input_ids": ids, "completion_mask": mask})
    if len(variants) != 1:
        sys.exit("ABORT: inconsistent chat_template_variant within %s: %s"
                 % (split, variants))
    if n_bos_bad:
        sys.exit("ABORT: %d examples in %s do not have exactly one BOS. "
                 "Run training/check_template.py --base <id>." % (n_bos_bad, split))
    stats = {"n": len(rows), "truncated": truncated,
             "variant": variants.pop(),
             "max_tokens": max(len(r["input_ids"]) for r in rows),
             "mean_tokens": round(sum(len(r["input_ids"]) for r in rows) / len(rows)),
             "mean_completion": round(sum(sum(r["completion_mask"]) for r in rows) / len(rows))}
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--qlora", action="store_true",
                    help="4-bit NF4 base (QLoRA) instead of a bf16/fp16 base (LoRA)")
    # 2, chosen from measurement rather than taste. 1 epoch is 75 optimizer
    # steps at effective batch 16, with only 2 warmup steps - likely too few to
    # move the model, which would produce a false negative across every arm at
    # once (the one failure mode that looks like a finding). 3 epochs projects
    # to ~13 h for the full matrix at the measured 18.1 s/step; 2 gives 150
    # steps for ~9 h. eval_loss is logged per epoch, so "was it still
    # improving when we stopped" is answerable from the record.
    ap.add_argument("--epochs", type=float, default=2.0)
    # 1280, not 1024. Measured on the real files with the real tokenizers:
    # train totals reach 1095 (Qwen) / 1139 (Mistral), and the longest example
    # has a 1111-token PROMPT - so a 1024 ceiling truncated its answer away
    # entirely, leaving an example with no trainable token. 4-5 train examples
    # exceed 1024 for both tokenizers; none exceeds 1280. Padding is dynamic
    # (the collator pads to the longest in each batch), so this raises the
    # worst-case batch cost, not the typical one: p95 total is 791/863.
    ap.add_argument("--max-length", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default 2*r")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=",".join(TARGET_MODULES),
                    help="comma-separated, or 'all-linear'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    # Checkpoint cadence. The original plan skipped resume entirely, on the
    # grounds that "225 steps is ~36 min for the longest arm, so restarting is
    # cheaper than the code". Measured throughput is 84 s/step, which makes one
    # arm 3.5 h and inverts that trade: a run killed at step 34 of 150 lost 48
    # minutes and left nothing on disk, because save_strategy="epoch" put the
    # first checkpoint at step 75. 25 steps is ~35 min of exposure.
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in the adapter dir")
    ap.add_argument("--name", default=None, help="adapter output name")
    a = ap.parse_args()

    method = "qlora" if a.qlora else "lora"
    tm = TARGET_MODULES if a.target_modules == ",".join(TARGET_MODULES) else (
        "all-linear" if a.target_modules == "all-linear"
        else [s.strip() for s in a.target_modules.split(",") if s.strip()])

    print("[1/5] dataset")
    assert_no_leakage()

    print("[2/5] hardware + seed")
    import torch
    if not torch.cuda.is_available():
        sys.exit("ABORT: no CUDA device visible. torch.cuda.is_available() is False.")
    bf16 = pick_precision(torch)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    try:
        import numpy as np
        np.random.seed(a.seed)
    except Exception:
        pass
    print("  seed: %d (python, numpy, torch)" % a.seed)

    print("[3/5] tokenizer + pre-tokenized data")
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(a.base)
    pad, how = pick_pad(tok)
    tok.pad_token = pad
    tok.padding_side = "right"
    print("  eos=%r(%s)  pad=%r(%s) via %s"
          % (tok.eos_token, tok.eos_token_id, tok.pad_token, tok.pad_token_id, how))
    if tok.pad_token_id == tok.eos_token_id:
        print("  WARNING: pad == eos. Masking here is driven by completion_mask and")
        print("           the attention mask, never by input_ids == pad_id, so real")
        print("           EOS targets are safe - but verify generations terminate.")

    tr_rows, tr_stats = build_pairs(tok, "train", a.max_length)
    va_rows, va_stats = build_pairs(tok, "val", a.max_length)
    print("  train: %s" % tr_stats)
    print("  val:   %s" % va_stats)
    if tr_stats["truncated"] or va_stats["truncated"]:
        print("  WARNING: %d train / %d val examples truncated at %d tokens"
              % (tr_stats["truncated"], va_stats["truncated"], a.max_length))
    ds_train, ds_val = Dataset.from_list(tr_rows), Dataset.from_list(va_rows)

    print("[4/5] model (%s)" % method)
    dtype = torch.bfloat16 if bf16 else torch.float16
    # device_map={"": 0} not "auto": with a 4-bit base on a single card, "auto"
    # can decide to offload a layer to CPU, which breaks training and needs
    # llm_int8_enable_fp32_cpu_offload. It also sets model_parallel, which
    # changes Trainer behaviour. This applies to the plain LoRA path too.
    kw = dict(dtype=dtype, device_map={"": 0})
    if a.qlora:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    model.config.use_cache = False
    if a.qlora:
        # use_reentrant must match what SFTConfig passes below: SFTConfig
        # (gradient_checkpointing=True) calls gradient_checkpointing_enable a
        # SECOND time, and a mismatch raises "none of the inputs have
        # requires_grad".
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    # No paged_adamw_8bit: adapter Adam state is a fraction of a GB so the
    # saving is noise, and bnb's paged optimizers need CUDA unified memory,
    # which Windows does not support.

    out_name = a.name or ("vjlm-%s-%s" % (method, a.base.split("/")[-1].lower()))
    out_dir = os.path.join(ADAPTERS, out_name)
    os.makedirs(out_dir, exist_ok=True)

    want = dict(output_dir=out_dir, num_train_epochs=a.epochs, max_steps=a.max_steps,
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
                # Must be set explicitly: it defaults to 8, and eval logits are
                # batch x seq x vocab x ~10 bytes. At 8 x 1280 x 151665 that is
                # ~15 GB on its own, which on Windows does not OOM - CUDA falls
                # back to host memory and evaluation crawls (measured: 10.7 s/it
                # and 34 GB "reserved" on a 1.5B model). Tie it to --batch so
                # the eval footprint matches the training footprint.
                per_device_eval_batch_size=a.batch,
                learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                logging_steps=10,
                # Step-wise, not per-epoch. Per-epoch yields 2 eval points and
                # places the first checkpoint 1h45m into a 3h30m run. Every 25
                # steps gives a 6-point curve - so "was it still improving when
                # we stopped" is answerable from the record rather than assumed
                # - and bounds the cost of a kill to ~35 min.
                #
                # No save_total_limit: 6 adapter checkpoints are ~2.5 GB for the
                # 7B arm against 45 GB free. Keeping them is what makes the eval
                # curve ACTIONABLE rather than merely descriptive - if val loss
                # turns, the earlier checkpoint still exists to fall back to.
                save_strategy="steps", save_steps=a.save_steps,
                eval_strategy="steps", eval_steps=a.eval_steps,
                bf16=bf16, fp16=not bf16,
                max_length=a.max_length, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                completion_only_loss=True, seed=a.seed, report_to="none")
    cfg = SFTConfig(**filter_kwargs(SFTConfig, want, {
        "max_length": "max_seq_length", "eval_strategy": "evaluation_strategy"}))

    tp = set(inspect.signature(SFTTrainer.__init__).parameters)
    tkw = {"processing_class": tok} if "processing_class" in tp else {"tokenizer": tok}

    print("[5/5] training  (effective batch %d = %d x %d)"
          % (a.batch * a.accum, a.batch, a.accum))
    t0 = time.time()
    # peft_config is passed to trl rather than calling get_peft_model here;
    # double-wrapping silently trains nothing.
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds_train,
                         eval_dataset=ds_val,
                         peft_config=LoraConfig(**lora_kwargs(
                             a.lora_r, a.lora_alpha, a.lora_dropout, tm)),
                         **tkw)
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")),
                   key=lambda q: int(q.rsplit("-", 1)[-1]))
    resumed_from = None
    if a.resume and ckpts:
        resumed_from = int(ckpts[-1].rsplit("-", 1)[-1])
        print("  resuming from %s (step %d of %s)"
              % (os.path.basename(ckpts[-1]), resumed_from, a.max_steps if a.max_steps > 0 else "?"))
        trainer.train(resume_from_checkpoint=ckpts[-1])
    else:
        if a.resume:
            print("  --resume given but no checkpoint under %s; starting fresh" % out_dir)
        trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    mins = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9

    # log_history is restored from the checkpoint on resume, so this is the
    # whole run's curve even when the run was killed and continued.
    hist = [h for h in trainer.state.log_history if "eval_loss" in h]
    best = min(hist, key=lambda h: h["eval_loss"]) if hist else None
    record = {
        "base_model": a.base, "method": method,
        "lora_r": a.lora_r, "lora_alpha": a.lora_alpha or 2 * a.lora_r,
        "lora_dropout": a.lora_dropout, "target_modules": tm,
        "epochs": a.epochs, "max_steps": a.max_steps,
        "batch": a.batch, "accum": a.accum, "effective_batch": a.batch * a.accum,
        "lr": a.lr, "max_length": a.max_length, "seed": a.seed,
        "precision": "bf16" if bf16 else "fp16",
        "pad_token": tok.pad_token, "pad_source": how,
        "chat_template_variant": tr_stats["variant"],
        "add_special_tokens": False,
        "train_stats": tr_stats, "val_stats": va_stats,
        "dataset_sha256": {s: sha256(os.path.join(DATA, s + ".jsonl"))
                           for s in ("train", "val", "test")},
        "peak_vram_gb": round(peak, 2), "peak_vram_reserved_gb": round(reserved, 2),
        "train_minutes": round(mins, 1),
        "eval_curve": [{"step": h.get("step"), "epoch": round(h.get("epoch", 0), 3),
                        "eval_loss": round(h["eval_loss"], 4)} for h in hist],
        "best_eval": ({"step": best.get("step"),
                       "eval_loss": round(best["eval_loss"], 4)} if best else None),
        "final_is_best": bool(best and hist and best is hist[-1]),
        # Wall-clock and peak VRAM cover THIS segment only. A resumed run does
        # not re-measure what the killed segment already did, so do not read
        # these as totals when resumed_from_step is set.
        "resumed_from_step": resumed_from,
        "git_sha": git_sha(), "pkg_versions": pkg_versions(),
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print("\ntrained in %.1f min | peak VRAM %.2f GB allocated, %.2f GB reserved"
          % (mins, peak, reserved))
    # Read these two numbers correctly, because Windows makes the naive reading
    # wrong in both directions:
    #
    #   allocated - live tensors at the high-water mark. This is the fit test.
    #   reserved  - everything the caching allocator ever held, including freed
    #               blocks it kept for reuse. Variable-length batches fragment
    #               it badly and `expandable_segments` is NOT supported on
    #               Windows, so reserved routinely exceeds device VRAM while
    #               the run is perfectly healthy (measured: 6.8 GB allocated
    #               vs 19.6 GB reserved on a 1.5B LoRA run that trained fine).
    #
    # And WDDM lets CUDA spill into shared system memory rather than raising
    # OOM, so a crash-free run is not proof it fitted - only `allocated` is.
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if peak > 0.9 * vram_gb:
        print("  WARNING: peak allocated (%.2f GB) is within 10%% of device VRAM"
              % peak)
        print("           (%.2f GB). On Windows this spills to host memory instead"
              % vram_gb)
        print("           of raising OOM, so timings stop being representative.")
        print("           Drop to --batch 1 --accum %d and record the deviation."
              % (a.batch * a.accum))
    print("eval curve (step: loss): %s" % "  ".join(
        "%s:%.4f" % (h["step"], h["eval_loss"]) for h in record["eval_curve"]))
    if record["best_eval"]:
        print("best eval_loss %.4f at step %s%s"
              % (record["best_eval"]["eval_loss"], record["best_eval"]["step"],
                 "" if record["final_is_best"] else
                 "  <- NOT the final step: val loss turned, so the final adapter "
                 "is over-fit relative to that checkpoint"))
    print("adapter -> %s" % out_dir)
    print("run record -> %s" % os.path.join(out_dir, "run.json"))
    print("\nNext: score BASE and this adapter with evaluation/eval_generation.py")
    print("      --backend hf. Without the base score the difference is not")
    print("      measurable (Phase 7 gate).")


if __name__ == "__main__":
    main()
