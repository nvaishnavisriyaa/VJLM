"""Score a model on the held-out VJLM behavioural test set.

Run once against the BASE model and once against each fine-tuned adapter, with
everything else identical. The difference is the only evidence that fine-tuning
helped. Without the base number, "the fine-tune worked" is unfalsifiable
(Phase 7 gate).

Two backends, one scorer (evaluation/metrics.py):

    hf    - in-process transformers. Loads the base (optionally 4-bit NF4) and
            attaches the LoRA adapter WITHOUT merging it. This is how Phase 8
            measures FT-001.
    http  - a running llama-server at $VJLM_LLAMA_URL. This is how Phase 9
            measures a quantized GGUF build in the deployment configuration.

Why Phase 8 does not measure a merged Q4_K_M build
--------------------------------------------------
A LoRA delta is small next to the weight magnitude it is added to. Q4_K_M
carries ~4.5 bits with per-block scales, so merged deltas below the
quantization step round away entirely. That is a floor effect pushing EVERY
arm toward "fine-tuning did not help" - and a null result caused by the
measurement is indistinguishable from a true null. Keeping the adapter
unmerged in bf16 on top of a quantized base means the trained delta is never
quantized at all. "Both sides get the same quantization" does not rescue the
merged design either: it controls the main effect but not a (model x quant)
interaction, and quantization damage is not uniform across architectures, so
the between-arm ranking can still skew. Quantization quality is QUANT-001's
job in Phase 9, measured against an f16 reference.

Usage:
    python evaluation/eval_generation.py --backend hf --base <hf-id> --qlora \
        --label mistral-base --arm B --role base
    python evaluation/eval_generation.py --backend hf --base <hf-id> --qlora \
        --adapter training/adapters/vjlm-qlora-... --label mistral-qlora \
        --arm B --role tuned
    python evaluation/eval_generation.py --compare
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.request, uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TEST = os.path.join(ROOT, "training", "data", "test.jsonl")
OUT = os.path.join(ROOT, "experiments", "generation_results.jsonl")
ARTIFACTS = os.path.join(ROOT, "experiments", "artifacts")
URL = os.environ.get("VJLM_LLAMA_URL", "http://127.0.0.1:8080")

from evaluation.metrics import aggregate, score_item  # noqa: E402
from training.check_template import apply_template, pick_pad  # noqa: E402
from training.train_local import git_sha, pkg_versions, sha256  # noqa: E402

MAX_TOKENS = 320
FAIL_ABORT_FRAC = 0.02


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def http_generator(retries=3):
    """llama-server. Retries, because a dropped request used to be counted in
    the denominator and silently deflated the score."""
    def gen(messages):
        body = json.dumps({"messages": messages, "max_tokens": MAX_TOKENS,
                           "temperature": 0.0, "seed": 0, "stream": False}).encode()
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    URL + "/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.loads(r.read())
                return d["choices"][0]["message"]["content"].strip(), None
            except Exception as e:                      # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        return None, str(last)[:160]
    return gen, {"serve_backend": "http", "serve_url": URL}


def hf_generator(base, adapter, qlora, seed, revision=None):
    """In-process transformers. The prompt is rendered exactly as
    training/train_local.py rendered it and as rag/retrieve.py sends it, so
    train, eval and serve all see the same string."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(seed)
    # A local directory has no revision; only pin when loading from the hub.
    # Before 2026-09-19 --revision was RECORDED in every row but never passed
    # here, so the pin was a claim rather than a guarantee. The cache happened
    # to hold the right commits, so the Phase 8 weights were correct - but an
    # unenforced provenance field is exactly the kind of thing this project
    # exists to not do.
    rev = None if os.path.isdir(base) else revision
    tok = AutoTokenizer.from_pretrained(adapter or base,
                                        **({} if (adapter or rev is None) else {"revision": rev}))
    pad, _ = pick_pad(tok)
    tok.pad_token = pad
    tok.padding_side = "right"

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    kw = dict(dtype=dtype, device_map={"": 0})
    if qlora:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    t0 = time.time()
    if rev:
        kw["revision"] = rev
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        # NOT merged: the adapter stays a separate bf16 low-rank path on top of
        # the quantized base, so the trained delta is never quantized.
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    load_s = time.time() - t0
    print("  loaded in %.1fs (4bit=%s, adapter=%s)"
          % (load_s, bool(qlora), os.path.basename(adapter) if adapter else None))

    variants = set()

    def gen(messages):
        text, variant = apply_template(tok, messages, add_generation_prompt=True)
        variants.add(variant)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_TOKENS,
                                 do_sample=False, temperature=None, top_p=None,
                                 top_k=None,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        new = out[0][ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), None

    meta = {"serve_backend": "hf", "infer_quant": "nf4" if qlora else
            ("bf16" if bf16 else "fp16"), "load_seconds": round(load_s, 1),
            "_variants": variants}
    return gen, meta


# --------------------------------------------------------------------------
def n_tokens(s):
    return len(re.findall(r"\S+", s))


KNOWN_ROLES = ("base", "tuned", "best", "l0-unmerged", "l0-merged", "l1-gguf")


def run(a):
    if a.role and not (a.role in KNOWN_ROLES or a.role.startswith("l2-")):
        print("  WARNING: unrecognised --role %r; recording anyway. "
              "Convention: base|tuned|best|l0-*|l1-gguf|l2-<QUANT>." % a.role)
    data = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    if a.limit:
        data = data[:a.limit]

    if a.backend == "hf":
        if not a.base:
            sys.exit("--backend hf needs --base")
        gen, meta = hf_generator(a.base, a.adapter, a.qlora, a.seed, a.revision)
    else:
        gen, meta = http_generator()

    run_id = uuid.uuid4().hex[:8]
    items, failures = [], []
    t0 = time.time()
    for i, ex in enumerate(data):
        msgs = ex["messages"]
        ref = msgs[-1]["content"]
        pred, err = gen(msgs[:-1])
        if err is not None:
            failures.append({"i": i, "error": err})
            print("  %d/%d FAILED: %s" % (i + 1, len(data), err), flush=True)
            continue
        rec = score_item(pred, ref, n_tokens(pred))
        rec["i"] = i
        rec["ref"] = ref
        items.append(rec)
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, len(data)), flush=True)
    dur = time.time() - t0

    res = aggregate(items, len(data), len(failures), seed=a.seed)
    if res["n_failed"] > FAIL_ABORT_FRAC * res["n_requested"]:
        print("\n%d of %d requests failed (> %.0f%%). The row is NOT recorded - a "
              "failure rate this high biases the comparison."
              % (res["n_failed"], res["n_requested"], FAIL_ABORT_FRAC * 100))
        sys.exit(1)

    os.makedirs(ARTIFACTS, exist_ok=True)
    per_item = os.path.join("experiments", "artifacts", "%s.jsonl" % run_id)
    with open(os.path.join(ROOT, per_item), "w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec) + "\n")
        for fl in failures:
            f.write(json.dumps({"failure": fl}) + "\n")

    variant = None
    if "_variants" in meta:
        v = meta.pop("_variants")
        variant = (v.pop() if len(v) == 1 else "MIXED:%s" % sorted(v))

    row = dict(res)
    row.update({
        "run_id": run_id, "label": a.label, "arm": a.arm, "role": a.role,
        "base_model": a.base, "base_revision": a.revision,
        "adapter": (os.path.basename(a.adapter.rstrip("/\\")) if a.adapter else None),
        "method": a.method or ("none" if not a.adapter else
                               ("qlora" if a.qlora else "lora")),
        "dataset": "test.jsonl", "dataset_sha256": sha256(TEST),
        "max_tokens": MAX_TOKENS, "temperature": 0.0, "seed": a.seed,
        "chat_template_variant": variant,
        "eval_sec_per_answer_gpu": round(dur / max(res["n_ok"], 1), 2),
        "deployment_sec_per_answer_cpu": None,
        "git_sha": git_sha(), "pkg_versions": pkg_versions(),
        "per_item": per_item, "ts": time.strftime("%Y-%m-%d %H:%M"),
    })
    row.update(meta)

    # Carry the training provenance across from the adapter's own record, so a
    # result row is self-contained without having to go and find run.json.
    if a.adapter:
        rj = os.path.join(a.adapter, "run.json")
        if os.path.exists(rj):
            tr = json.load(open(rj, encoding="utf-8"))
            for k in ("lora_r", "lora_alpha", "target_modules", "epochs",
                      "effective_batch", "max_length", "peak_vram_gb",
                      "train_minutes", "eval_loss_by_epoch"):
                row.setdefault(k, tr.get(k))

    print("\n" + json.dumps({k: v for k, v in row.items()
                             if k not in ("pkg_versions",)}, indent=2))
    # A --limit run is a smoke test, not a measurement. Recording it would put
    # a row scored on 3 items next to rows scored on 147 in the same file, and
    # --compare has no way to know the difference.
    if a.limit:
        print("--limit %d set: NOT recorded to %s (partial run)."
              % (a.limit, os.path.basename(OUT)))
        print("per-item -> %s" % per_item)
        return
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("appended -> %s" % OUT)
    print("per-item -> %s" % per_item)
    if res["n_failed"]:
        print("NOTE: %d request(s) failed and were excluded from the denominator."
              % res["n_failed"])


def compare():
    """Key on the configuration, never on the label.

    The previous version did `by[row["label"]] = row`, so two runs sharing a
    label silently overwrote each other, last write wins. With a five-arm
    matrix that loses a run without any warning.
    """
    if not os.path.exists(OUT):
        sys.exit("no results yet: %s does not exist" % OUT)
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    by = {}
    for r in rows:
        key = (r.get("base_model"), r.get("method"), r.get("lora_r"),
               r.get("infer_quant"), r.get("role"), r.get("adapter"))
        prev = by.get(key)
        if prev is None or r.get("ts", "") >= prev.get("ts", ""):
            by[key] = r          # newest wins, and duplicates are reported
    dupes = len(rows) - len(by)

    keys = ["n_ok", "n_failed", "grounded_f1", "grounded_f1_ci95",
            "abstain_recall", "abstain_precision", "false_abstain",
            "citation_rate", "mean_answer_tokens", "eval_sec_per_answer_gpu",
            "peak_vram_gb", "train_minutes"]
    order = sorted(by.values(), key=lambda r: (r.get("arm") or "", r.get("role") or ""))
    hdr = ["%s/%s" % (r.get("arm") or "?", r.get("role") or "?") for r in order]
    print("%-26s %s" % ("", "  ".join("%-16s" % h for h in hdr)))
    print("%-26s %s" % ("model", "  ".join(
        "%-16s" % ((r.get("base_model") or "?").split("/")[-1][:16]) for r in order)))
    print("%-26s %s" % ("method", "  ".join(
        "%-16s" % ("%s%s" % (r.get("method"), "" if not r.get("lora_r")
                             else " r%s" % r["lora_r"])) for r in order)))
    print("-" * (26 + 18 * len(order)))
    for k in keys:
        print("%-26s %s" % (k, "  ".join(
            "%-16s" % ("-" if r.get(k) is None else r.get(k)) for r in order)))

    print("\nbase -> tuned deltas, each arm against its OWN base:")
    tuned = [r for r in by.values() if r.get("role") in ("tuned", "best")]
    bases = {r.get("base_model"): r for r in by.values() if r.get("role") == "base"}
    for r in sorted(tuned, key=lambda x: x.get("arm") or ""):
        b = bases.get(r.get("base_model"))
        if not b:
            print("  arm %s: NO BASE RESULT for %s - delta not computable"
                  % (r.get("arm"), r.get("base_model")))
            continue
        d = lambda k: (None if r.get(k) is None or b.get(k) is None
                       else round(r[k] - b[k], 4))
        print("  arm %-3s %-28s f1 %+0.4f  abst_rec %s  cite %+0.4f  len %s tok"
              % (r.get("arm"), (r.get("base_model") or "").split("/")[-1][:28],
                 d("grounded_f1") or 0.0,
                 d("abstain_recall"), d("citation_rate") or 0.0,
                 d("mean_answer_tokens")))
    if dupes:
        print("\n%d superseded row(s) in the file were ignored (same configuration)."
              % dupes)
    print("\nNOTE: scored on templated held-out examples, not human questions."
          "\n      Phase 5 (evaluation/questions.jsonl) is still empty, so these"
          "\n      numbers measure BEHAVIOUR, not real answer quality.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "http"], default="hf")
    ap.add_argument("--base", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--qlora", action="store_true",
                    help="load the base in 4-bit NF4 (must match the arm)")
    ap.add_argument("--method", default=None, help="none|lora|qlora")
    ap.add_argument("--arm", default=None, help="A|B|C")
    # "best" is the lowest-eval_loss checkpoint rather than the final step.
    # Both completed arms bottom out inside epoch 1 and then rise, and
    # trainer.save_model() writes the FINAL step - so "tuned" is the over-fit
    # side of the curve. Scoring "best" as its own row makes "does one epoch
    # beat two" a measurement instead of an argument from eval_loss, which is
    # token-level cross-entropy and not the behaviour being graded.
    # l0-* and l1-* are Phase 9 reference levels, not Phase 8 arms. They exist
    # so merge error, GGUF conversion error and quantization error stay
    # separable instead of being summed into one number:
    #   l0-unmerged  adapter on a bf16 base, unquantized  - the true fine-tune
    #   l0-merged    merged bf16 weights, unquantized     - isolates MERGE error
    #   l1-gguf      BF16 GGUF through llama-server       - isolates CONVERSION
    # Free-form, with a documented convention, because Phase 9 needs one role
    # per quantized build and a fixed choices list would have to grow every
    # time the ladder does. Unknown values warn rather than abort - the role is
    # a provenance label, and refusing to record a completed 147-item run over
    # a naming quibble would waste the run.
    #   Phase 8:  base | tuned | best
    #   Phase 9:  l0-unmerged  adapter on bf16 base, unquantized
    #             l0-merged    merged bf16 weights   -> isolates MERGE error
    #             l1-gguf      BF16 GGUF             -> isolates CONVERSION error
    #             l2-<QUANT>   a quantized build     -> isolates QUANTIZATION
    ap.add_argument("--role", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        return compare()
    run(a)


if __name__ == "__main__":
    main()
