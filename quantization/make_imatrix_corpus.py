"""Build a VJLM-representative calibration corpus for llama-imatrix.

An importance matrix records which weights actually carry activation on text
the model will really see, so the quantizer can spend its limited bits there.
That only helps if the calibration text resembles deployment traffic - a
generic wiki dump would optimise for the wrong distribution.

What goes in
------------
Two sources, both already committed, so the corpus is reproducible from the
repo rather than from whatever happened to be on disk:

  * preprocessing/out/rag_chunks.jsonl - the retrieved passages that dominate
    every real prompt. This is the bulk of the token budget.
  * training/data/train.jsonl - rendered through the model's OWN chat template,
    so the special tokens and role headers that wrap every deployed request are
    represented. Without these, the imatrix never sees <|start_header_id|> and
    friends in position, and llama-imatrix is invoked with --parse-special for
    the same reason.

Only the TRAIN split is used. The val and test splits stay untouched: letting
calibration see held-out text would leak the benchmark into the artifact being
benchmarked.

What does NOT go in
-------------------
Hindi and Telugu. The request asked for them, but the corpus contains zero
Devanagari across all 1,493 train/val/test rows and 4 of 13,051 RAG chunks
contain any Telugu. There is nothing representative to calibrate on and no
eval set to verify against, so including token noise would be worse than
honest omission. Recorded in QUANT-001.

Usage:
    python quantization/make_imatrix_corpus.py --model llama --out corpus.txt
"""
import argparse, json, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CHUNKS = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")
TRAIN = os.path.join(ROOT, "training", "data", "train.jsonl")

BASES = {
    "llama": ("meta-llama/Llama-3.2-3B-Instruct",
              "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3",
                "c170c708c41dac9275d15a8fff4eca08d52bab71"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(BASES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=1400,
                    help="RAG passages to sample (the retrieval-shaped bulk)")
    ap.add_argument("--dialogs", type=int, default=400,
                    help="training dialogs rendered in the chat template")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from training.check_template import apply_template

    base, rev = BASES[a.model]
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    rng = random.Random(a.seed)

    parts, n_chunk_tok, n_dlg_tok = [], 0, 0

    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    rng.shuffle(rows)
    for r in rows[:a.chunks]:
        t = (r.get("text") or "").strip()
        if len(t) > 200:
            parts.append(t)
            n_chunk_tok += len(tok(t, add_special_tokens=False)["input_ids"])

    dlgs = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    rng.shuffle(dlgs)
    for d in dlgs[:a.dialogs]:
        # Rendered exactly as training and serving render it, so the imatrix
        # sees the real prompt scaffolding rather than bare prose.
        text, _ = apply_template(tok, d["messages"], add_generation_prompt=False)
        parts.append(text)
        n_dlg_tok += len(tok(text, add_special_tokens=False)["input_ids"])

    rng.shuffle(parts)
    body = "\n\n".join(parts)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(body)

    total = n_chunk_tok + n_dlg_tok
    print("corpus -> %s" % a.out)
    print("  %d RAG passages   %7d tokens (%.0f%%)"
          % (min(a.chunks, len(rows)), n_chunk_tok, 100.0 * n_chunk_tok / max(total, 1)))
    print("  %d chat dialogs   %7d tokens (%.0f%%)"
          % (min(a.dialogs, len(dlgs)), n_dlg_tok, 100.0 * n_dlg_tok / max(total, 1)))
    print("  %d total tokens under the %s tokenizer, %.1f MB on disk"
          % (total, a.model, os.path.getsize(a.out) / 1e6))
    print("\nRun llama-imatrix with --parse-special so the chat-template tokens\n"
          "are counted as tokens and not as literal text.")


if __name__ == "__main__":
    main()
