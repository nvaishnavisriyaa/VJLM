"""Where does a miss actually happen - recall, or ranking?

For every question whose answer is NOT in the top 5, report where the gold chunk
sits in each stage:

  bm25_rank    position in the full BM25 ranking (the candidate pool is drawn from it)
  pool         was it a candidate at all
  final_rank   position in the final ranking (not truncated to 5)

A gold chunk at final_rank 6-20 is a RANKING problem - weights can fix it.
A gold chunk missing from the pool is a RECALL problem - no weighting can fix it.

    python evaluation/diagnose_misses.py [mode]
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402
from rag.index import tok                      # noqa: E402


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "coverage"
    r = R.Retriever()
    print("chunks: %d" % len(r.meta), flush=True)
    recall, ranking, fine = 0, 0, 0
    for qid, split, kind, q, allx, src in Q1 + Q2:
        gold = [i for i, m in enumerate(r.meta)
                if (not src or re.search(src, m["url"], re.I))
                and all(re.search(a, m["text"], re.I) for a in allx)]
        top5 = r.search(q, role="student", topk=5, mode=mode)
        if any(all(re.search(a, c["text"], re.I) for a in allx)
               and (not src or re.search(src, c["url"], re.I)) for c in top5):
            fine += 1
            continue
        if not gold:
            print("%-4s NO-GOLD-CHUNK  %s" % (qid, q[:60]))
            continue
        deep = r.search(q, role="student", topk=200, mode=mode)
        keys = [(c["url"], c["text"][:60]) for c in deep]
        gk = {(r.meta[i]["url"], r.meta[i]["text"][:60]) for i in gold}
        frank = next((n + 1 for n, k in enumerate(keys) if k in gk), None)
        allsc = r.bm25.search_weighted([(w, 1.0) for w in tok(q)], topk=100000)
        pos = {i: n + 1 for n, (i, _) in enumerate(allsc)}
        brank = min((pos.get(i, 10 ** 9) for i in gold), default=None)
        if frank:
            ranking += 1
        else:
            recall += 1
        print("%-4s %-5s final=%-5s bm25=%-7s golds=%-3d %s" % (
            qid, split, frank, brank if brank and brank < 10 ** 9 else "none",
            len(gold), q[:58]))
    print("\nhit@5 %d | ranking problems (gold in deep ranking) %d | recall problems %d"
          % (fine, ranking, recall))


if __name__ == "__main__":
    main()
