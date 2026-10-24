"""Does the answer survive into the prompt TEXT, not just into the top 5?

evaluation/hit_at_prompt.py checks the gold patterns against each chunk's FULL
text. The model never sees that. build_prompt gives each of the 5 chunks
PROMPT_CHARS // 5 characters and rag/focus.py keeps the sentences that match
the question - so a chunk can be selected, hold the answer, and still reach the
model with the answer cut out. The answer is often the part of the passage the
question's words do NOT appear in, which is exactly what focus drops.

Retrieval does not depend on PROMPT_CHARS, so this retrieves once and rebuilds
the prompts at each budget.

    python evaluation/hit_in_prompt_text.py [topk] [chars,chars,...]
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402


def ok(text, url, allx, src):
    if src and not re.search(src, url, re.I):
        return False
    return all(re.search(a, text, re.I) for a in allx)


def main():
    topk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    budgets = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2
                                else "3250,5000,6500,8000").split(",")]
    # Only override when asked. Defaulting to a different weighting here than
    # rag/retrieve.py uses is how the app came to serve an unmeasured config.
    if os.environ.get("VJLM_FUSION"):
        R.FUSION = tuple(float(x) for x in os.environ["VJLM_FUSION"].split(","))
    r = R.Retriever()
    qs = Q1 + Q2

    got = []
    for qid, split, kind, q, allx, src in qs:
        got.append((qid, q, allx, src, r.search(q, role="student", topk=topk, mode="fusion")))

    # Ceiling: the answer is in a selected chunk's full text.
    full = sum(1 for _, _, a, s, res in got
               if any(ok(c["text"], c["url"], a, s) for c in res))
    print("topk=%d  chunks selected: answer in full chunk text  %d/%d = %.0f%%"
          % (topk, full, len(qs), 100 * full / len(qs)))

    for b in budgets:
        R.PROMPT_CHARS = b
        hits = 0
        lost = []
        for qid, q, allx, src, res in got:
            msgs = R.build_prompt(q, res, role="student")
            body = msgs[1]["content"]
            # Split the context back into its numbered passages so a pattern
            # cannot be satisfied by two different passages at once.
            passages = re.split(r"\n\n(?=\[\d+\] )", body.split("CONTEXT:\n", 1)[-1])
            if any(ok(p, p.split("\n", 1)[0], allx, src) for p in passages):
                hits += 1
            else:
                lost.append(qid)
        shown = len(R.build_prompt(got[0][1], got[0][4], role="student")[1]["content"])
        print("  PROMPT_CHARS=%-5d per chunk %-4d  in prompt text %d/%d = %.0f%%   "
              "(user msg %d chars ~%d tok)"
              % (b, max(220, b // topk), hits, len(qs), 100 * hits / len(qs),
                 shown, shown // 4))
        print("     trimmed away: %s" % " ".join(lost))


if __name__ == "__main__":
    main()
