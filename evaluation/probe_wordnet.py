"""What the IDF gate keeps and what it throws away, on the real corpus.

The gate's claim is that wrong WordNet senses are COMMON words and useful ones
are RARE, so idf(synonym) >= ratio * idf(term) separates them. This prints the
decision for each candidate so the claim can be checked by eye before any
Hit@5 number is trusted.

    python evaluation/probe_wordnet.py [ratio ...]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from rag import retrieve as R          # noqa: E402
from rag import synonyms as S          # noqa: E402

GOOD = ["start", "begin", "failed", "fail", "rule", "exam", "recheck",
        "lowest", "orientation", "seats", "short"]
BAD = ["miss", "club", "seat", "apply", "attend", "do", "get", "year"]


def main():
    ratios = [float(x) for x in sys.argv[1:]] or [0.0, 0.8, 1.0, 1.2]
    r = R.Retriever()
    idf = r.expander.vocab
    print("vocab is %s with %d entries\n" % (type(idf).__name__, len(idf)))

    for ratio in ratios:
        S.WN_IDF_RATIO = ratio
        S._raw_synonyms.cache_clear()
        print("=" * 72)
        print("ratio %.1f  (keep synonym if idf(syn) >= %.1f x idf(term))" % (ratio, ratio))
        for label, terms in (("want kept", GOOD), ("want dropped", BAD)):
            print("-- %s" % label)
            for t in terms:
                ti = idf.get(t)
                raw = S._raw_synonyms(t)
                kept = r.expander.expand_term(t)
                drops = []
                for c in raw:
                    if c in kept:
                        continue
                    si = idf.get(c)
                    if si is None:
                        drops.append("%s(absent)" % c)
                    else:
                        drops.append("%s(%.1f)" % (c, si))
                print("  %-11s idf=%-5s keep=%-34s drop=%s"
                      % (t, ("%.1f" % ti) if ti else "-",
                         ", ".join(kept) or "-", ", ".join(drops[:6]) or "-"))
        print()


if __name__ == "__main__":
    main()
