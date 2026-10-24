"""Can an embedding FIRST STAGE reach the chunks BM25 never surfaces?

rag/retrieve.py builds its candidate pool from the BM25 ranking, so coverage and
cosine only ever re-rank what keyword matching already found. Measured with
evaluation/diagnose_misses.py, 14 of 25 misses are chunks BM25 ranks 300-9000th:
no weighting can reach them. This asks the other question - where does the gold
chunk sit in the COSINE ranking over the whole corpus - which decides whether a
dense first stage is worth adding.

    python evaluation/dense_recall.py
"""
import os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402
from rag.index import tok                      # noqa: E402

PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    r = R.Retriever()
    if r.dense is None:
        print("no dense index"); return
    buckets = [10, 50, 100, 200, 500, 2000]
    tally = {b: 0 for b in buckets}
    tally["worse"] = 0
    n = 0
    for qid, split, kind, q, allx, src in Q1 + Q2:
        gold = [i for i, m in enumerate(r.meta)
                if (not src or re.search(src, m["url"], re.I))
                and all(re.search(a, m["text"], re.I) for a in allx)]
        if not gold:
            continue
        n += 1
        qv = np.asarray(list(r.embed.embed([PREFIX + q]))[0], dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-9)
        sims = r.dense @ qv
        order = np.argsort(-sims)
        pos = {int(i): k + 1 for k, i in enumerate(order[:max(buckets)])}
        best = min((pos.get(i, 10 ** 9) for i in gold), default=10 ** 9)
        placed = False
        for b in buckets:
            if best <= b:
                tally[b] += 1
                placed = True
                break
        if not placed:
            tally["worse"] += 1
        # BM25 position, for the side-by-side
        allsc = r.bm25.search_weighted([(w, 1.0) for w in tok(q)], topk=100000)
        bpos = {i: k + 1 for k, (i, _) in enumerate(allsc)}
        bbest = min((bpos.get(i, 10 ** 9) for i in gold), default=10 ** 9)
        print("%-4s cosine=%-7s bm25=%-7s %s" % (
            qid, best if best < 10 ** 9 else ">2000",
            bbest if bbest < 10 ** 9 else "none", q[:52]), flush=True)
    print("\ngold chunk's best COSINE rank, %d questions:" % n)
    run = 0
    for b in buckets:
        run += tally[b]
        print("  within top %-5d %3d  (%3.0f%% cumulative)" % (b, tally[b], 100 * run / n))
    print("  beyond 2000      %3d" % tally["worse"])


if __name__ == "__main__":
    main()
