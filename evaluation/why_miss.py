"""Which ARM is responsible for each miss.

Three guesses at new signals (cross-encoder, best-sentence, overview-page
bonus) were all rejected by measurement, so stop guessing: for every question
whose answer is below rank 5, print the gold chunk's score broken down by arm
next to the chunk that outranked it. If the gold chunk loses on `sem`, the
embedding is backing the wrong passage; if it loses on `kw`, the question's
words genuinely are not in it; if it loses only on the role/family multiplier,
a prior is at fault, not the arms.

    python evaluation/why_miss.py [deep]
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


def line(tag, c, p):
    return "  %-5s %5.3f = kw %5.3f + syn %5.3f + sem %5.3f  (kwraw %.2f bon %.2f semraw %.2f) %s" % (
        tag, c["score"], p.get("kw", 0), p.get("syn", 0), p.get("sem", 0),
        p.get("raw_kw", 0), p.get("bonus", 0), p.get("sem_raw", 0),
        (c["url"] or "")[-40:])


def main():
    deep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    # Only override when asked. Defaulting to a different weighting here than
    # rag/retrieve.py uses is how the app came to serve an unmeasured config.
    if os.environ.get("VJLM_FUSION"):
        R.FUSION = tuple(float(x) for x in os.environ["VJLM_FUSION"].split(","))
    r = R.Retriever()
    lose = {"kw": 0, "syn": 0, "sem": 0, "prior": 0}
    for qid, split, kind, q, allx, src in Q1 + Q2:
        R.PARTS = {}
        res = r.search(q, role="student", topk=deep, mode="fusion")
        parts = R.PARTS
        R.PARTS = None
        rank = next((i + 1 for i, c in enumerate(res) if ok(c, allx, src)), None)
        if (rank or 10 ** 9) <= 5:
            continue
        print("\n%-4s rank=%-5s %s" % (qid, rank or ">%d" % deep, q[:60]))
        if not rank:
            continue
        g, t = res[rank - 1], res[0]
        gi = next(i for i, m in enumerate(r.meta) if m is g or m["chunk_id"] == g["chunk_id"])
        ti = next(i for i, m in enumerate(r.meta) if m["chunk_id"] == t["chunk_id"])
        pg, pt = parts.get(gi, {}), parts.get(ti, {})
        print(line("top", t, pt))
        print(line("gold", g, pg))
        # which arm accounts for most of the gap
        gaps = {k: pt.get(k, 0) - pg.get(k, 0) for k in ("kw", "syn", "sem")}
        worst = max(gaps, key=gaps.get)
        if max(gaps.values()) <= 0:
            worst = "prior"
        lose[worst] += 1
        print("  gap: " + "  ".join("%s %+.3f" % (k, v) for k, v in gaps.items())
              + "   -> %s" % worst)
    print("\nlost mainly on: " + "  ".join("%s %d" % kv for kv in lose.items()))


if __name__ == "__main__":
    main()
