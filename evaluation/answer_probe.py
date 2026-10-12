"""End-to-end answer probe: retrieval -> prompt -> llama-server -> attribution.

Runs without the Flask app (so no background pipeline competes for the CPU) and
prints the final answer exactly as the UI would show it, plus quality checks:
  - boilerplate: the learned "check this against your own regulation batch" hedge
  - copied:      answer mostly a verbatim paste of one passage
  - unsupported: sentences no retrieved passage backs
  - has_must:    required facts present (the conditions a complete answer needs)
Usage: python evaluation/answer_probe.py [--mode coverage]
"""
import argparse, os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from rag.retrieve import Retriever, build_prompt  # noqa: E402
from rag.answer import attribute, with_supplement, used_from  # noqa: E402
import app.server as srv  # noqa: E402  (llama_chat only; refresher is not started here)

CASES = [
    ("What is the minimum attendance required to write semester end exams in B.Tech R25?",
     [r"75\s*%", r"65\s*%", r"condon"]),
    ("How do I pay my college fees online and which cards are not allowed?",
     [r"rupay", r"(UPI|Internet Banking)", r"(Easypay|website)"]),
    ("What happens to a B.Tech R25 student whose shortage of attendance is not condoned?",
     [r"detain", r"re-?admission|same semester"]),
    ("How many marks does CIE carry and how many does SEE carry in R25 B.Tech?",
     [r"40", r"60"]),
    ("What is the highest package in the 2026 placement season?",
     [r"65\s*LPA|65 lakh"]),
    ("What are my internal marks this semester?", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="coverage")
    a = ap.parse_args()
    r = Retriever()
    for q, must in CASES:
        t0 = time.time()
        ch = r.search(q, role="student", topk=5, mode=a.mode)
        raw = srv.llama_chat(build_prompt(q, ch, "student")) if ch else srv.ABSTAIN
        final, used, uns = attribute(raw, ch)
        final = with_supplement(q, final, ch, r.bm25.idf)
        used = used_from(final, len(ch))
        print("=" * 78)
        print("Q:", q)
        print("-" * 78)
        print(final)
        flags = []
        if re.search(r"regulation batch|differ(s)? (across|between)", raw, re.I):
            flags.append("BOILERPLATE")
        if re.match(r"\s*in simple terms", raw, re.I):
            flags.append("IN-SIMPLE-TERMS")
        if uns:
            flags.append("UNSUPPORTED=%d" % len(uns))
        if must is None:
            flags.append("ABSTAINED-OK" if "could not find" in final.lower() else "SHOULD-HAVE-ABSTAINED")
        else:
            miss = [m for m in must if not re.search(m, final, re.I)]
            flags.append("ALL-FACTS" if not miss else "MISSING:" + "|".join(miss))
        print("\n[%s] used=%s  %.0fs" % (" ".join(flags), [i + 1 for i in used], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
