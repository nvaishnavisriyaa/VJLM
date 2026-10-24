"""Hit rate at increasing depth: separates recall from last-mile ranking.

If hit@50 is high but hit@5 is not, the right chunks ARE candidates and the
weights/ordering are the problem. If hit@50 is also low, the pool itself misses
them and no reranking can help.

    python evaluation/depth_curve.py "cov;1,0.5,1.5"
"""
import os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402

DEPTHS = (1, 3, 5, 10, 20, 50, 100)


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def main():
    grid = (sys.argv[1] if len(sys.argv) > 1 else "cov;1,0.5,1.5").split(";")
    r = R.Retriever()
    qs = Q1 + Q2
    print("%-12s %s" % ("kw,syn,sem", "  ".join("@%-4d" % d for d in DEPTHS)), flush=True)
    for g in grid:
        mode = "coverage" if g.strip() in ("cov", "coverage") else "fusion"
        if mode == "fusion":
            R.FUSION = tuple(float(x) for x in g.split(","))
        t0, ranks = time.time(), []
        for qid, split, kind, q, allx, src in qs:
            top = r.search(q, role="student", topk=max(DEPTHS), mode=mode)
            ranks.append(next((i + 1 for i, c in enumerate(top) if hit(c, allx, src)), None))
        n = len(ranks)
        cells = ["%3.0f%%" % (100 * sum(1 for x in ranks if x and x <= d) / n) for d in DEPTHS]
        print("%-12s %s   %.0fs" % (g, "  ".join("%-5s" % c for c in cells), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
