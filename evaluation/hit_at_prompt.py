"""hit@5 as the PROMPT actually sees it, with the per-document caps binding.

evaluation/depth_curve.py retrieves once at depth 100 and slices, which is right
for measuring recall but blind to the caps: they scale with topk (`grow =
topk // 5`), so at depth 100 they never bind. Every cap setting therefore
measured identical there - and the end-to-end test disagreed, because a
5-passage prompt is exactly where a cap throws passages away. n17's answer sits
at overall rank 5 and never reaches the prompt: the top ten are all clauses of
the R25 rulebook, and a cap of 2 per document replaces clauses 3-5 with worse
documents.

This asks for topk passages the way the app does, so what it reports is what the
model is shown.

    python evaluation/hit_at_prompt.py [topk]
"""
import os, re, sys
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
    topk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    # Only override when asked. Defaulting to a different weighting here than
    # rag/retrieve.py uses is how the app came to serve an unmeasured config.
    if os.environ.get("VJLM_FUSION"):
        R.FUSION = tuple(float(x) for x in os.environ["VJLM_FUSION"].split(","))
    r = R.Retriever()
    qs = Q1 + Q2
    ok = miss = 0
    missed = []
    for qid, split, kind, q, allx, src in qs:
        res = r.search(q, role="student", topk=topk, mode="fusion")
        if any(hit(c, allx, src) for c in res):
            ok += 1
        else:
            miss += 1
            missed.append(qid)
    print("topk=%d  cap_frac=%s cap_min=%s  hit %d/%d = %.0f%%" % (
        topk, R.DOC_CAP_FRAC, R.DOC_CAP_MIN, ok, len(qs), 100 * ok / len(qs)))
    print("missed: %s" % " ".join(missed))


if __name__ == "__main__":
    main()
