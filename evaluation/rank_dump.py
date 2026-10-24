"""Per-question rank of the first hit, as JSON, for comparing two knowledge bases.

    python evaluation/rank_dump.py out.json [mode]

Point it at a candidate knowledge base with VJLM_OUT_DIR / VJLM_INDEX_DIR.
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def main():
    out = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "coverage"
    r = R.Retriever()
    res = {}
    for qid, split, kind, q, allx, src in Q1 + Q2:
        top = r.search(q, role="student", topk=5, mode=mode)
        rank = next((i + 1 for i, c in enumerate(top) if hit(c, allx, src)), None)
        # does ANY chunk of the knowledge base satisfy the pattern (so a miss is not a data loss)?
        res[qid] = {"rank": rank, "q": q, "split": split,
                    "top1": (top[0]["url"][-40:] + " | " + top[0]["text"][:60].replace("\n", " ")) if top else ""}
    json.dump(res, open(out, "w"), indent=0)
    n = len(res)
    print("%s: hit@5 %.0f%%  hit@1 %.0f%%  (n=%d)" % (
        out, 100 * sum(1 for v in res.values() if v["rank"]) / n,
        100 * sum(1 for v in res.values() if v["rank"] == 1) / n, n))


if __name__ == "__main__":
    main()
