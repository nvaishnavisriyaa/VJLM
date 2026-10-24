"""Tune the three retrieval weights: keyword, synonym (WordNet), semantic (embedding).

Same questions and hit rule as eval_student_retrieval.py. Weights are CHOSEN on
the dev split; test and hold are reported so over-fitting to dev shows.

  python evaluation/tune_fusion.py                       # default grid
  python evaluation/tune_fusion.py --grid "1,.5,0;1,.5,.5"
"""
import argparse, os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1  # noqa: E402
from student_questions_v2 import Q2  # noqa: E402
from rag import retrieve as R    # noqa: E402

DEFAULT = ("cov;1,0.5,0;1,0.5,0.15;1,0.5,0.3;1,0.5,0.5;1,0.5,0.8;1,0.5,1.2;"
           "1,0.25,0.5;1,1,0.5")


def hit(chunk, allx, src):
    if src and not re.search(src, chunk["url"], re.I):
        return False
    return all(re.search(a, chunk["text"], re.I) for a in allx)


def run(r, which, mode="fusion", topk=5):
    by = {}
    qs = (Q1 if which in ("v1", "both") else []) + (Q2 if which in ("v2", "both") else [])
    for qid, split, kind, q, allx, src in qs:
        res = r.search(q, role="student", topk=topk, mode=mode)
        rank = next((i + 1 for i, c in enumerate(res) if hit(c, allx, src)), None)
        for k in ("ALL", split, kind):
            by.setdefault(k, []).append(rank)
    return by


def fmt(rs):
    n = len(rs)
    h5 = sum(1 for x in rs if x) / n
    h1 = sum(1 for x in rs if x == 1) / n
    mrr = sum(1 / x for x in rs if x) / n
    return "%3.0f%% %3.0f%% %.3f" % (100 * h5, 100 * h1, mrr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=DEFAULT)
    ap.add_argument("--set", default="both", choices=["v1", "v2", "both"])
    ap.add_argument("--pool", type=int, default=None, help="semantic candidates added to the pool")
    ap.add_argument("--qprefix", action="store_true", help="BGE search-query instruction prefix")
    a = ap.parse_args()
    if a.pool:
        R.FUSION_DENSE_POOL = a.pool
    if a.qprefix:
        R.FUSION_QPREFIX = "Represent this sentence for searching relevant passages: "
    r = R.Retriever()
    print("dense loaded:", r.dense is not None, " semantic pool:", R.FUSION_DENSE_POOL, flush=True)
    print("%-14s %-17s %-17s %-17s %-17s  (hit@5 hit@1 MRR)" %
          ("kw,syn,sem", "ALL", "dev", "test", "hold"), flush=True)
    for g in a.grid.split(";"):
        mode = "fusion"
        if g.strip() in ("cov", "coverage"):
            mode = "coverage"
        else:
            R.FUSION = tuple(float(x) for x in g.split(","))
        t = time.time()
        by = run(r, a.set, mode)
        print("%-14s %-17s %-17s %-17s %-17s  %.0fs" % (
            g, fmt(by["ALL"]), fmt(by["dev"]), fmt(by["test"]), fmt(by["hold"]),
            time.time() - t), flush=True)


if __name__ == "__main__":
    main()
