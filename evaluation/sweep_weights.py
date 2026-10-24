"""Find the three weights by search instead of by hand.

1.0 / 0.5 / 1.5 was set by hand. Only the RATIOS matter for ranking, so keyword
is pinned at 1.0 and the other two are swept against it.

Tuned on dev+hold, reported on test+v2-unseen. That split is the point: the 62
questions in student_questions.py were written while tuning retrieval, so a
weight chosen on them and reported on them measures nothing. The 36 v2
questions (n01-n36) have never been tuned against.

    python evaluation/sweep_weights.py
"""
import os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402

SYN = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
SEM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def main():
    r = R.Retriever()
    qs = [(q[0], ("tune" if q[1] in ("dev", "hold") else "report"), q[3], q[4], q[5])
          for q in Q1]
    qs += [(q[0], "unseen", q[3], q[4], q[5]) for q in Q2]
    groups = {}
    for qid, g, q, allx, src in qs:
        groups.setdefault(g, 0)
        groups[g] += 1
    print("questions: %s  (weights chosen on 'tune' only)\n"
          % " ".join("%s=%d" % kv for kv in sorted(groups.items())))

    rows = []
    for sy in SYN:
        for se in SEM:
            R.FUSION = (1.0, sy, se)
            t0 = time.time()
            got = {"tune": 0, "report": 0, "unseen": 0}
            for qid, g, q, allx, src in qs:
                res = r.search(q, role="student", topk=5, mode="fusion")
                if any(hit(c, allx, src) for c in res):
                    got[g] += 1
            tot = sum(got.values())
            rows.append((sy, se, got, tot, time.time() - t0))
            print("W=%.2f S=%.2f   tune %2d/%-2d  report %2d/%-2d  UNSEEN %2d/%-2d   "
                  "all %2d/%d = %3.0f%%   %.0fs"
                  % (sy, se, got["tune"], groups["tune"], got["report"],
                     groups["report"], got["unseen"], groups["unseen"],
                     tot, len(qs), 100 * tot / len(qs), time.time() - t0))

    best = max(rows, key=lambda x: (x[2]["tune"], x[3]))
    sy, se, got, tot, _ = best
    print("\nbest on tune split: W=%.2f S=%.2f" % (sy, se))
    print("  tune   %2d/%d = %.0f%%" % (got["tune"], groups["tune"],
                                        100 * got["tune"] / groups["tune"]))
    print("  report %2d/%d = %.0f%%  (held out from the choice)"
          % (got["report"], groups["report"], 100 * got["report"] / groups["report"]))
    print("  unseen %2d/%d = %.0f%%  (never tuned against at all)"
          % (got["unseen"], groups["unseen"], 100 * got["unseen"] / groups["unseen"]))
    print("  all    %2d/%d = %.0f%%" % (tot, len(qs), 100 * tot / len(qs)))

    top = sorted(rows, key=lambda x: -x[3])[:5]
    print("\nbest overall (for reference - NOT how a weight should be chosen):")
    for sy, se, got, tot, _ in top:
        print("  W=%.2f S=%.2f  all %2d/%d = %3.0f%%  unseen %2d/%d"
              % (sy, se, tot, len(qs), 100 * tot / len(qs),
                 got["unseen"], groups["unseen"]))


if __name__ == "__main__":
    main()
