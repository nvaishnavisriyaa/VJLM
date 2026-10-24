"""The current misses, with the gold chunk's deep rank and what beat it.

Fast (no full-corpus BM25 pass): retrieves once at `deep`, so it says whether a
miss is reachable (gold inside `deep`) and what outranked it.

    python evaluation/miss_report.py [deep]
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402


def ok(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def flat(s, n=66):
    return re.sub(r"\s+", " ", s)[:n].encode("ascii", "replace").decode()


def main():
    deep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    # Only override when asked. Defaulting to a different weighting here than
    # rag/retrieve.py uses is how the app came to serve an unmeasured config.
    if os.environ.get("VJLM_FUSION"):
        R.FUSION = tuple(float(x) for x in os.environ["VJLM_FUSION"].split(","))
    r = R.Retriever()
    miss = 0
    for qid, split, kind, q, allx, src in Q1 + Q2:
        res = r.search(q, role="student", topk=deep, mode="fusion")
        rank = next((i + 1 for i, c in enumerate(res) if ok(c, allx, src)), None)
        if rank and rank <= 5:
            continue
        miss += 1
        print("\n%-4s %-5s rank=%-5s %s" % (qid, split, rank or ">%d" % deep, q[:62]))
        for c in res[:3]:
            print("     %-46s %s" % (c["url"][-46:], flat(c.get("context") or c["text"])))
        if rank:
            g = res[rank - 1]
            print("  GOLD %-46s %s" % (g["url"][-46:], flat(g.get("context") or g["text"])))
    print("\n%d misses of %d" % (miss, len(Q1 + Q2)))


if __name__ == "__main__":
    main()
