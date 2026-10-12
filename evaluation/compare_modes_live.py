"""Ask the live app the same questions through every retrieval mode.

Qualitative side-by-side, not a benchmark - there are no gold labels here.

EVERY QUESTION IS GROUNDED IN VERIFIED CORPUS CONTENT (checked 2026-09-30).
The first version of this file asked about attendance, fees and scholarships.
All six modes refused all four questions, which looked like a system failure
and was not: the corpus contains 0 chunks stating an attendance rule, 8 chunks
of fees/scholarships, and its 'academic_regulations' category is 32 syllabus
booklets, not the rulebook. The policy PDFs live on docs.vnrvjiet.ac.in, which
robots.txt excludes (INGEST-002, BLOCKED). Asking unanswerable questions makes
all modes behave identically and measures nothing.

  * literal      - uses the corpus's own vocabulary. Lexical modes should win.
  * paraphrased  - student wording, answer present under other words.
  * bridging     - 'bullying' must reach 'ragging'. Neither word is a WordNet
                   synonym of the other, so this is the case that separates
                   embeddings from every lexical mode.
  * unanswerable - genuinely absent. Correct response is refusal.
"""
import json
import sys
import time
import urllib.request

APP = "http://127.0.0.1:7860/api/ask"
MODES = ["bm25", "coverage", "lexical", "dense", "hybrid", "hybrid3"]

# Each question is followed by the evidence that it IS answerable.
QUESTIONS = [
    # II-B.Tech-CSE-R25.pdf p2: "25PC1CS205 Design and Analysis of Algorithms 3 0 0 3 3"
    ("literal", "How many credits is the Design and Analysis of Algorithms course in B.Tech CSE R25?"),
    # same table, III SEMESTER R25 course list
    ("paraphrased", "what subjects will I be studying in third semester CSE"),
    # anti-ragging-cell p1: "Zero tolerance for ragging and substance abuse on campus"
    ("bridging", "who do I report to if someone is bullying me on campus"),
    # no student-specific records anywhere in public data
    ("unanswerable", "What are my internal marks this semester?"),
]


def ask(q, mode, role="student", timeout=600):
    body = json.dumps({"q": q, "role": role, "mode": mode}).encode()
    req = urllib.request.Request(APP, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    d["_wall"] = round(time.time() - t0, 1)
    return d


def abstained(text):
    t = (text or "").lower()
    return "could not find" in t or "not find this" in t


def main():
    only = sys.argv[1:] or MODES
    rows = []
    for kind, q in QUESTIONS:
        print("\n" + "=" * 78)
        print("[%s] %s" % (kind.upper(), q))
        print("=" * 78)
        for mode in only:
            try:
                d = ask(q, mode)
            except Exception as e:                      # noqa: BLE001
                print("\n-- %-9s REQUEST FAILED: %s" % (mode, str(e)[:90]))
                continue
            ans = (d.get("answer") or "").strip()
            cites = d.get("citations", [])
            ab = abstained(ans)
            ok = (ab if kind == "unanswerable" else not ab)
            rows.append({"kind": kind, "q": q, "mode": mode,
                         "abstained": ab, "correct_behaviour": ok,
                         "n_cites": len(cites),
                         "t_retrieve": d.get("t_retrieve"),
                         "t_gen": d.get("t_gen"), "answer": ans,
                         "citations": [c.get("url", "").rsplit("/", 1)[-1] for c in cites]})
            print("\n-- %-9s  retrieve %-6s gen %-6s %s"
                  % (mode, d.get("t_retrieve"), d.get("t_gen"),
                     "ABSTAINED" if ab else ""))
            print("   %s" % ans[:400].replace("\n", "\n   "))
            for c in cites[:3]:
                print("      [%s p%s] %s" % (c.get("category"), c.get("page"),
                                             c.get("url", "").rsplit("/", 1)[-1][:46]))

    out = "experiments/mode_comparison_live.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 78)
    print("BEHAVIOUR SUMMARY  (correct = answered when answerable, refused when not)")
    print("=" * 78)
    print("%-10s %-11s %-13s %-10s %-13s %s"
          % ("mode", "literal", "paraphrased", "bridging", "unanswerable", "abstain rate"))
    for mode in only:
        mr = [r for r in rows if r["mode"] == mode]
        if not mr:
            continue
        def frac(kind):
            k = [r for r in mr if r["kind"] == kind]
            return "%d/%d" % (sum(r["correct_behaviour"] for r in k), len(k)) if k else "-"
        ar = sum(r["abstained"] for r in mr) / len(mr)
        print("%-10s %-11s %-13s %-10s %-13s %.0f%%"
              % (mode, frac("literal"), frac("paraphrased"), frac("bridging"),
                 frac("unanswerable"), 100 * ar))
    print("\nwrote -> %s" % out)


if __name__ == "__main__":
    main()
