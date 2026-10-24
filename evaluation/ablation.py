"""The A-E ablation: what each of the three signals is actually worth.

Reports Hit@5 (at least one chunk holding the answer in the 5 the prompt gets)
and, from a second deep pass, where the answer sits when it is not in the 5.
That second number is the diagnostic that decides what to work on next:

  answer in the candidate pool but below rank 5  -> a RANKING problem, the
      weights and the ordering can still fix it
  answer not in the pool at all                  -> a RECALL problem, no
      weighting will ever reach it

Hit@5 must be measured at topk=5, NOT by slicing a deep ranking: the per-
document caps scale with topk (`grow = topk // 5`), so at depth 1200 they never
bind and the top 5 of a deep run is not the top 5 the app serves. Each config
therefore runs twice - once at 5 for the real number, once deep for recall.

Only the RATIOS of the weights affect ranking, so normalising them to sum to 1
cannot change any result; it is presentation. Config E is included because its
ratios genuinely differ from D (semantic 0.67x keyword, not 1.5x).

    python evaluation/ablation.py [deep]        deep: pool depth, default 1200
"""
import os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402

# label, (keyword, synonym, semantic), wordnet, domain, what it tests
CONFIGS = [
    ("A  keyword only",          (1.0, 0.0, 0.0), True,  True),
    ("B  keyword+semantic",      (1.0, 0.0, 1.5), True,  True),
    ("C  keyword+synonym",       (1.0, 0.5, 0.0), True,  True),
    ("D  all three",             (1.0, 0.5, 1.5), True,  True),
    ("E  all three, sum-to-1",   (0.45, 0.25, 0.30), True, True),
    ("D- wordnet off (domain)",  (1.0, 0.5, 1.5), False, True),
    ("D- domain off (wordnet)",  (1.0, 0.5, 1.5), True,  False),
]


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def rank_of(res, allx, src):
    for n, c in enumerate(res, 1):
        if hit(c, allx, src):
            return n
    return None


def main():
    deep = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    r = R.Retriever()
    qs = Q1 + Q2
    rows = []
    for label, w, use_wn, use_dom in CONFIGS:
        R.FUSION = w
        r.expander.use_wordnet = use_wn
        r.expander.use_domain = use_dom
        for attr in ("_cache", "cache", "_memo"):
            if isinstance(getattr(r.expander, attr, None), dict):
                getattr(r.expander, attr).clear()

        t0 = time.time()
        at5 = 0
        ranks = []
        for qid, split, kind, q, allx, src in qs:
            res = r.search(q, role="student", topk=5, mode="fusion")
            if any(hit(c, allx, src) for c in res):
                at5 += 1
                ranks.append(1)
                continue
            # Not in the 5 the prompt gets. Where is it?
            deep_res = r.search(q, role="student", topk=deep, mode="fusion")
            ranks.append(rank_of(deep_res, allx, src))
        dt = time.time() - t0

        n = len(qs)
        in_pool = sum(1 for x in ranks if x is not None)
        band = lambda lo, hi: sum(1 for x in ranks if x is not None and lo <= x <= hi)
        rows.append((label, w, at5, in_pool, band(6, 20), band(21, 100),
                     band(101, deep), n - in_pool, dt / n))
        print("%-26s K=%.2f W=%.2f S=%.2f  hit@5 %2d/%d = %3.0f%%   "
              "in pool %3.0f%%  |  6-20:%2d  21-100:%2d  101+:%2d  absent:%2d  "
              "%.2fs/q"
              % (label, w[0], w[1], w[2], at5, n, 100 * at5 / n,
                 100 * in_pool / n, band(6, 20), band(21, 100),
                 band(101, deep), n - in_pool, dt / n))

    print("\n%-26s %7s %8s %9s" % ("", "Hit@5", "in pool", "headroom"))
    for label, w, at5, in_pool, b1, b2, b3, absent, _ in rows:
        n = len(qs)
        print("%-26s %6.0f%% %7.0f%% %8.0f%%"
              % (label, 100 * at5 / n, 100 * in_pool / n,
                 100 * (in_pool - at5) / n))
    print("\nheadroom = reachable by ranking alone (in the pool, below rank 5)")


if __name__ == "__main__":
    main()
