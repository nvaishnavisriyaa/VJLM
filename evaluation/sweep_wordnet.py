"""Make the WordNet arm earn its weight, and prove it with Hit@5.

The arm measured as worth exactly 0 (83% with it, 83% without). Three reasons,
all configuration rather than principle:

  POS       nouns only, so the verbs the questions actually turn on - start,
            fail, recheck - expanded to nothing.
  cap       max_per_term=3, and DOMAIN candidates are queued before WordNet
            ones, so the hand-built map fills every slot and WordNet is
            crowded out even when it has the right lemma.
  senses    with verbs allowed back in, general-English noise returns
            (miss -> girl), which is what the IDF gate is for.

Swept together because they interact: lifting the cap without the gate buys
noise, and the gate does nothing while the cap is already full.

Tuned on dev+hold, reported separately on test and on the 36 never-tuned v2
questions, because a synonym choice made and scored on the same questions
measures nothing.

    python evaluation/sweep_wordnet.py
"""
import os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402
from rag import synonyms as S                  # noqa: E402

POS = ["n", "nv", "nva"]
RATIO = [0.0, 1.0, 1.3]
CAP = [3, 6]
SYNW = [0.5, 1.0]


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def reset(r):
    S._raw_synonyms.cache_clear()
    r.expander.last = {}
    for obj in (r.expander, r.coverage):
        for attr in ("_cache", "cache", "_memo", "_terms"):
            v = getattr(obj, attr, None)
            if isinstance(v, dict):
                v.clear()
    for fn in ("build_terms",):
        f = getattr(r.coverage, fn, None)
        if hasattr(f, "cache_clear"):
            f.cache_clear()


def main():
    r = R.Retriever()
    qs = [(q[0], ("tune" if q[1] in ("dev", "hold") else "test"), q[3], q[4], q[5])
          for q in Q1]
    qs += [(q[0], "unseen", q[3], q[4], q[5]) for q in Q2]
    sizes = {}
    for _, g, _, _, _ in qs:
        sizes[g] = sizes.get(g, 0) + 1
    print("questions: %s\n" % " ".join("%s=%d" % kv for kv in sorted(sizes.items())))

    rows = []
    for pos in POS:
        for ratio in RATIO:
            for cap in CAP:
                for sw in SYNW:
                    S.WN_POS, S.WN_IDF_RATIO = pos, ratio
                    r.expander.max_per_term = cap
                    R.FUSION = (1.0, sw, 1.5)
                    reset(r)
                    t0 = time.time()
                    got = {"tune": 0, "test": 0, "unseen": 0}
                    for qid, g, q, allx, src in qs:
                        res = r.search(q, role="student", topk=5, mode="fusion")
                        if any(hit(c, allx, src) for c in res):
                            got[g] += 1
                    tot = sum(got.values())
                    rows.append((pos, ratio, cap, sw, got, tot))
                    print("POS=%-3s gate=%.1f cap=%d W=%.1f | tune %2d/%-2d "
                          "test %2d/%-2d UNSEEN %2d/%-2d | all %2d/%d = %3.0f%% %.0fs"
                          % (pos, ratio, cap, sw, got["tune"], sizes["tune"],
                             got["test"], sizes["test"], got["unseen"],
                             sizes["unseen"], tot, len(qs),
                             100 * tot / len(qs), time.time() - t0))

    print("\n--- ranked by the TUNE split (the only legitimate selector)")
    for pos, ratio, cap, sw, got, tot in sorted(rows, key=lambda x: -x[4]["tune"])[:8]:
        print("POS=%-3s gate=%.1f cap=%d W=%.1f  tune %2d  test %2d  unseen %2d  all %2d/%d = %.0f%%"
              % (pos, ratio, cap, sw, got["tune"], got["test"], got["unseen"],
                 tot, len(qs), 100 * tot / len(qs)))
    print("\n--- ranked by the UNSEEN split (the honest generalisation check)")
    for pos, ratio, cap, sw, got, tot in sorted(rows, key=lambda x: -x[4]["unseen"])[:8]:
        print("POS=%-3s gate=%.1f cap=%d W=%.1f  tune %2d  test %2d  unseen %2d  all %2d/%d = %.0f%%"
              % (pos, ratio, cap, sw, got["tune"], got["test"], got["unseen"],
                 tot, len(qs), 100 * tot / len(qs)))


if __name__ == "__main__":
    main()
