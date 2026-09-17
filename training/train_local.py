"""Phase 8 — LoRA fine-tune, run locally on a CUDA GPU.

The notebooks under training/notebooks/ target Colab: /content/ paths, Drive
checkpoints, and a free T4. This runs the same FT-001 configuration on a local
card, so Colab is no longer in the loop (FOUND-003 allowed Colab because no
local CUDA GPU existed; that is no longer true).

Precision is DETECTED, not assumed. HANDOFF.md 4 says "use fp16 on a T4"
because Turing has no native bf16 and emulates it - that is a T4 workaround,
not a general rule. On Ampere or newer, bf16 is correct and more stable. The
check below picks the right one on whichever machine this runs on.

Per experiments/PLAN.md, Qwen2.5-1.5B is the CONTROL arm, not the selected
model - it exists to prove the training path end to end. The base model is
chosen at Phase 7 from a measured comparison, not from this default.

Usage:
    python training/train_local.py                    # control run
    python training/train_local.py --base <hf-id> --epochs 1
"""
import argparse, inspect, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "training", "data")
ADAPTERS = os.path.join(ROOT, "training", "adapters")

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# FT-001: LoRA r=16, alpha=32, dropout=0.05, all attention + MLP projections.
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--name", default=None, help="adapter output name")
    a = ap.parse_args()

    print("[1/4] dataset")
    assert_no_leakage()

    print("[2/4] hardware")
    import torch
    if not torch.cuda.is_available():
        sys.exit("ABORT: no CUDA device visible. torch.cuda.is_available() is False.")
    bf16 = pick_precision(torch)

    print("[3/4] model + data")
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files={
        "train": os.path.join(DATA, "train.jsonl"),
        "validation": os.path.join(DATA, "val.jsonl")})
    ds = ds.map(lambda b: {"text": [tok.apply_chat_template(m, tokenize=False)
                                    for m in b["messages"]]},
                batched=True, remove_columns=["messages"])

    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, device_map="auto")
    model.config.use_cache = False

    out_name = a.name or ("vjlm-lora-" + a.base.split("/")[-1].lower())
    out_dir = os.path.join(ADAPTERS, out_name)
    os.makedirs(out_dir, exist_ok=True)

    want = dict(output_dir=out_dir, num_train_epochs=a.epochs,
                per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
                learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                logging_steps=10, save_strategy="epoch", save_total_limit=1,
                eval_strategy="epoch", bf16=bf16, fp16=not bf16,
                max_length=a.max_length, gradient_checkpointing=True,
                report_to="none")
    cfg = SFTConfig(**filter_kwargs(SFTConfig, want, {
        "max_length": "max_seq_length", "eval_strategy": "evaluation_strategy"}))

    tp = set(inspect.signature(SFTTrainer.__init__).parameters)
    tkw = {"processing_class": tok} if "processing_class" in tp else {"tokenizer": tok}

    print("[4/4] training")
    t0 = time.time()
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=ds["validation"], peft_config=LoraConfig(**LORA), **tkw)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    mins = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 1e9

    print("\ntrained in %.1f min | peak VRAM %.1f GB" % (mins, peak))
    print("adapter -> %s" % out_dir)
    print("\nNext: merge and quantize, then score BASE and FINE-TUNED with\n"
          "      evaluation/eval_generation.py. Without the base score the\n"
          "      difference is not measurable (Phase 7 gate).")


if __name__ == "__main__":
    main()
