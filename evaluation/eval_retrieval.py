"""Compare retrieval variants on a VJLM benchmark: BM25 vs dense vs hybrid.

Builds a held-out benchmark by sampling distinctive passages from the real
corpus and forming a query from each, with the source document as the gold
answer. Then scores every retrieval variant on the SAME questions.

LIMITATION, stated honestly: these questions are auto-derived from passages,
so they favour lexical overlap and are a proxy for real student questions.
They are sufficient to rank retrieval variants against each other, which is
what this experiment is for. Human-written questions are still required for
the Phase 5 benchmark before any claim about answer quality.
"""
import json, os, random, re, sys, time, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
random.seed(7)

CHUNKS = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")
BENCH = os.path.join(ROOT, "evaluation", "retrieval_benchmark.jsonl")

STOPish = set("the of and to in for a an is are was were be on at by with as from this that it its "
              "shall will may can not no if then than which who what when where how".split())


def make_query(text):
    """A question-like query from a passage's distinctive terms."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 60 < len(s.strip()) < 260]
    if not sents:
        return None
    s = random.choice(sents)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", s) if w.lower() not in STOPish]
    if len(words) < 6:
        return None
    return " ".join(words[:12])


def build(n=180):
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    by_cat = defaultdict(list)
    for r in rows:
        if len(r["text"]) > 400:
            by_cat[r["category"]].append(r)
    # balance across categories so one huge category cannot dominate the score
    cats = [c for c in by_cat if c != "faculty" and len(by_cat[c]) >= 5]
    per = max(3, n // max(len(cats), 1))
    out = []
    for c in cats:
        for r in random.sample(by_cat[c], min(per, len(by_cat[c]))):
            q = make_query(r["text"])
            if q:
                out.append({"question": q, "gold_source_id": r["source_id"],
                            "gold_chunk_id": r["chunk_id"], "category": c,
                            "role": random.choice(["student", "faculty"])})
    random.shuffle(out)
    with open(BENCH, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("built %d benchmark questions across %d categories -> %s" % (len(out), len(cats), BENCH))
    return out


def score(retr, bench, mode, k=5, expand=False):
    """Recall@k and MRR at the DOCUMENT level (did we find the right source?)."""
    hits = 0
    rr = 0.0
    lat = []
    for b in bench:
        t0 = time.time()
        res = retr.search(b["question"], role=b["role"], topk=k, mode=mode,
                          expand=expand)
        lat.append(time.time() - t0)
        ids = [c["source_id"] for c in res]
        if b["gold_source_id"] in ids:
            hits += 1
            rr += 1.0 / (ids.index(b["gold_source_id"]) + 1)
    n = len(bench)
    lat.sort()
    return {"mode": mode + ("+syn" if expand else ""),
            "recall@%d" % k: round(hits / n, 3), "mrr": round(rr / n, 3),
            "p50_ms": round(lat[n // 2] * 1000, 1), "p95_ms": round(lat[int(n * 0.95)] * 1000, 1),
            "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    if a.build or not os.path.exists(BENCH):
        bench = build()
    else:
        bench = [json.loads(l) for l in open(BENCH, encoding="utf-8")]
        print("loaded %d benchmark questions" % len(bench))

    from rag.retrieve import Retriever
    r = Retriever()
    base_modes = ["bm25", "coverage", "lexical"]
    if r.dense is not None:
        base_modes += ["dense", "hybrid", "hybrid3"]
    # Expansion only touches the lexical arm, so a dense-only run would be
    # identical with it on - testing it would just record a duplicate row.
    # Coverage without expansion degrades to exact-match coverage, which is a
    # meaningful control: it isolates what the SYNONYMS add from what the
    # coverage MEASURE adds.
    modes = [(m, False) for m in base_modes] +             [(m, True) for m in base_modes if m != "dense"]
    if r.dense is None:
        print("NOTE: dense vectors not built - comparing BM25 only. "
              "Run rag/index.py to enable dense and hybrid.")

    print("\n%-8s %-10s %-7s %-9s %-9s" % ("mode", "recall@%d" % a.k, "mrr", "p50_ms", "p95_ms"))
    results = []
    for m, xp in modes:
        s = score(r, bench, m, a.k, expand=xp)
        results.append(s)
        print("%-10s %-10s %-7s %-9s %-9s" % (s["mode"], s["recall@%d" % a.k], s["mrr"], s["p50_ms"], s["p95_ms"]))

    out = os.path.join(ROOT, "experiments", "retrieval_results.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        for s in results:
            s["ts"] = time.strftime("%Y-%m-%d %H:%M")
            f.write(json.dumps(s) + "\n")
    if len(results) > 1:
        best = max(results, key=lambda s: s["mrr"])
        print("\nBEST by MRR: %s" % best["mode"])
    print("appended ->", out)


if __name__ == "__main__":
    main()
