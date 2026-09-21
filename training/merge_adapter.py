"""Merge a trained LoRA/QLoRA adapter into its base model, in BF16.

This is the first step of Phase 9 (QUANT-001): llama.cpp consumes a single set
of weights, not a base plus an adapter, so the adapter has to be folded in
before any GGUF conversion. Merging is NOT retraining - no gradient is computed
and no data is seen. It is a weight arithmetic step: W <- W + (alpha/r) * B @ A.

Precision
---------
The output is **BF16**, which is what both arms actually trained in
(`run.json: precision`). It is deliberately NOT downcast to FP16 here.

BF16 and FP16 are different numerical representations - same width, different
exponent/mantissa split - so BF16 -> FP16 is a lossy re-rounding, not a no-op.
If an FP16 build is ever needed (e.g. because a llama.cpp CPU backend handles
it better), it gets produced as its own artifact and the BF16 -> FP16 delta is
MEASURED, per the L1/L1b split in the Phase 9 plan. It is never assumed free.

A note on the QLoRA arms
------------------------
Arms B and C trained their adapters against an NF4-quantized base, while this
script merges into the original full-precision weights. That is the standard
deployment path, and merging itself introduces no quantization - but the
adapter was optimized against a dequantized-NF4 forward pass, so its deltas are
tuned to slightly different weights than the ones they land on. Any behavioural
difference from that mismatch is caught by the L0 gate: the merged model is
re-scored on the same 147 items and must reproduce the adapter's published
number within noise before anything downstream runs.

Verification
------------
"It ran without error" is not evidence the merge happened. `merge_and_unload()`
silently no-ops if the adapter did not attach to anything - for instance if the
target modules do not match the loaded architecture. So this script snapshots a
sample of weights before and after and asserts:

  * every sampled TARGET module changed (q_proj, k_proj, ... - these carry LoRA)
  * a sampled NON-target module did not change (embed_tokens - a negative
    control; if this moved, something other than the adapter was applied)

Usage:
    python training/merge_adapter.py --adapter vjlm-lora-llama-3.2-3b
    python training/merge_adapter.py --adapter vjlm-qlora-mistral-7b-v0.3
"""
import argparse, json, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ADAPTERS = os.path.join(ROOT, "training", "adapters")
OUT_ROOT = os.path.join(ROOT, "quantization", "builds", "merged")

from training.train_local import git_sha, pkg_versions, sha256  # noqa: E402

# Pinned revisions, identical to training/run_matrix.py. "we used Mistral-7B"
# is not a reproducible claim; main moves.
REVISIONS = {
    "meta-llama/Llama-3.2-3B-Instruct": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}

# Sampled for the before/after check. The first four carry LoRA on every arm;
# the last is the negative control and must NOT move.
PROBE_TARGET = ["model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.15.mlp.down_proj.weight",
                "model.layers.15.self_attn.o_proj.weight"]
PROBE_CONTROL = "model.embed_tokens.weight"

# Windows degrades badly under ~5 GB free, and a half-written 14 GB model is
# worse than a refusal.
MIN_FREE_GB = 8.0


def need_gb(n_params_b):
    """BF16 weights plus a little room for the shard being written."""
    return n_params_b * 2.0 * 1.15


def snapshot(model, names):
    out = {}
    sd = dict(model.named_parameters())
    for n in names:
        for key in (n, "base_model.model." + n):
            if key in sd:
                t = sd[key].detach()
                out[n] = (float(t.float().norm()), tuple(t.shape))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="directory name under training/adapters/")
    ap.add_argument("--checkpoint", default=None,
                    help="merge a checkpoint-N subdir instead of the final adapter")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing merged directory")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    src = os.path.join(ADAPTERS, a.adapter)
    if a.checkpoint:
        src = os.path.join(src, a.checkpoint)
    if not os.path.isdir(src):
        sys.exit("ABORT: no such adapter directory: %s" % src)

    cfg_path = os.path.join(src, "adapter_config.json")
    if not os.path.exists(cfg_path):
        sys.exit("ABORT: %s has no adapter_config.json" % src)
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    base_id = cfg["base_model_name_or_path"]
    rev = REVISIONS.get(base_id)
    if rev is None:
        sys.exit("ABORT: no pinned revision recorded for %s. Add one to "
                 "REVISIONS rather than silently resolving main." % base_id)

    # run.json sits beside the final adapter (not inside checkpoint dirs).
    rec_path = os.path.join(ADAPTERS, a.adapter, "run.json")
    rec = json.load(open(rec_path, encoding="utf-8")) if os.path.exists(rec_path) else {}

    name = a.adapter + ("-" + a.checkpoint if a.checkpoint else "") + "-bf16"
    out_dir = a.out or os.path.join(OUT_ROOT, name)
    if os.path.exists(out_dir):
        if not a.force:
            sys.exit("ABORT: %s already exists. Pass --force to overwrite, but "
                     "check first whether something downstream is using it." % out_dir)
        shutil.rmtree(out_dir)

    free_gb = shutil.disk_usage(ROOT).free / 2**30
    print("[1/6] pre-flight")
    print("  adapter      %s" % src)
    print("  base         %s @ %s" % (base_id, rev[:12]))
    print("  method       %s  r=%s alpha=%s" % (rec.get("method", "?"),
                                                cfg.get("r"), cfg.get("lora_alpha")))
    print("  target mods  %s" % ", ".join(sorted(cfg.get("target_modules", []))))
    print("  out          %s" % out_dir)
    print("  disk free    %.1f GB" % free_gb)
    if free_gb < MIN_FREE_GB:
        sys.exit("ABORT: only %.1f GB free, need >= %.1f. Windows degrades badly "
                 "below ~5 GB and a truncated model is worse than no model."
                 % (free_gb, MIN_FREE_GB))

    print("[2/6] loading base in bf16 (NOT quantized - this is the "
          "full-precision merge target)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_id, revision=rev, dtype=torch.bfloat16, device_map="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print("  %.1fB params in %s, loaded in %.1fs"
          % (n_params / 1e9, next(model.parameters()).dtype, time.time() - t0))

    want = need_gb(n_params / 1e9)
    if free_gb < want + 2:
        sys.exit("ABORT: need ~%.1f GB for the merged model, %.1f GB free."
                 % (want, free_gb))

    print("[3/6] attaching adapter")
    before = snapshot(model, PROBE_TARGET + [PROBE_CONTROL])
    model = PeftModel.from_pretrained(model, src, dtype=torch.bfloat16)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    if n_lora == 0:
        sys.exit("ABORT: adapter attached 0 LoRA parameters. The target modules "
                 "in adapter_config.json do not match this architecture.")
    print("  %d LoRA parameters attached" % n_lora)

    print("[4/6] merging (W <- W + (alpha/r) B@A)")
    t0 = time.time()
    model = model.merge_and_unload()
    print("  merged in %.1fs" % (time.time() - t0))

    print("[5/6] verifying the merge actually changed weights")
    after = snapshot(model, PROBE_TARGET + [PROBE_CONTROL])
    report, bad = [], []
    for n in PROBE_TARGET:
        if n not in before or n not in after:
            continue
        b, aa = before[n][0], after[n][0]
        rel = abs(aa - b) / max(b, 1e-9)
        report.append((n, b, aa, rel, "changed" if rel > 1e-6 else "UNCHANGED"))
        if rel <= 1e-6:
            bad.append(n)
    cb, ca = before.get(PROBE_CONTROL), after.get(PROBE_CONTROL)
    ctrl_rel = None
    if cb and ca:
        ctrl_rel = abs(ca[0] - cb[0]) / max(cb[0], 1e-9)
        report.append((PROBE_CONTROL, cb[0], ca[0], ctrl_rel,
                       "unchanged (control OK)" if ctrl_rel <= 1e-6
                       else "CHANGED - control violated"))
    for n, b, aa, rel, verdict in report:
        print("  %-46s %12.4f -> %12.4f  rel %.2e  %s" % (n[-46:], b, aa, rel, verdict))
    if bad:
        sys.exit("ABORT: %d target module(s) did not change. merge_and_unload() "
                 "no-opped - the adapter did not attach where it should have."
                 % len(bad))
    if ctrl_rel is not None and ctrl_rel > 1e-6:
        sys.exit("ABORT: the non-target control tensor changed. Something other "
                 "than the adapter was applied to these weights.")

    print("[6/6] saving bf16 + tokenizer + chat template")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(src if os.path.exists(
        os.path.join(src, "tokenizer_config.json")) else base_id, revision=None
        if os.path.exists(os.path.join(src, "tokenizer_config.json")) else rev)
    tok.save_pretrained(out_dir)
    for extra in ("chat_template.jinja",):
        s = os.path.join(src, extra)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(out_dir, extra))

    adapter_file = os.path.join(src, "adapter_model.safetensors")
    meta = {
        "merged_from_adapter": a.adapter,
        "checkpoint": a.checkpoint,
        "adapter_sha256": sha256(adapter_file) if os.path.exists(adapter_file) else None,
        "base_model": base_id,
        "base_revision": rev,
        "method": rec.get("method"),
        "lora_r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": sorted(cfg.get("target_modules", [])),
        "merged_dtype": "bfloat16",
        "n_params": n_params,
        "n_lora_params_attached": n_lora,
        "training_precision": rec.get("precision"),
        "train_run_ts": rec.get("ts"),
        "probe": {n: {"before": b, "after": aa, "rel": rel}
                  for n, b, aa, rel, _ in report},
        "git_sha": git_sha(), "pkg_versions": pkg_versions(),
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    json.dump(meta, open(os.path.join(out_dir, "merge.json"), "w", encoding="utf-8"),
              indent=1)

    size_gb = sum(os.path.getsize(os.path.join(out_dir, f))
                  for f in os.listdir(out_dir)) / 2**30
    print("\nmerged -> %s  (%.2f GB bf16)" % (out_dir, size_gb))
    print("record -> %s" % os.path.join(out_dir, "merge.json"))
    print("\nNext (L0 gate): score this merged model on the SAME 147 items and\n"
          "confirm it reproduces the adapter's published number within noise,\n"
          "BEFORE converting to GGUF. If the merge is wrong, every downstream\n"
          "measurement is of the wrong model.")


if __name__ == "__main__":
    main()
