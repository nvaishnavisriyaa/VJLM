"""ANSWER accuracy over all 98 corpus-verified questions, through the live app.

hit@5 is a proxy: it says the answer was among the passages, not that the model
said it. The 26-question test (evaluation/student_accuracy_coverage.py) measures
the real thing but is too small to support a "90%" claim - one question is four
points. This asks the same question of the 62 tuned questions plus the 36 that
were written from the corpus and never tuned on, and reports the splits
separately so a development figure is never mistaken for a held-out one.

The gold patterns are the ones the retrieval sets already carry, verified by
evaluation/check_questions.py to match real corpus text. A question counts as
answered when every pattern appears in the answer (`strict`, the same bar
retrieval is held to) and, more leniently, when any does (`any`).

    python evaluation/answer_accuracy.py [mode] [out.jsonl]

The passage count is the APP's (VJLM_TOPK), not this script's, so restart the
app to change it.
"""
import json, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402

URL = "http://127.0.0.1:7860/api/ask"
ABSTAIN = "could not find this"


def ask(q, mode):
    body = json.dumps({"question": q, "q": q, "query": q,
                       "role": "student", "mode": mode}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fusion"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "experiments/answer_accuracy.jsonl"
    qs = Q1 + Q2
    rows = []
    out = open(out_path, "w", encoding="utf-8")
    for n, (qid, split, kind, q, allx, src) in enumerate(qs, 1):
        t = time.time()
        try:
            d = ask(q, mode)
        except Exception as e:
            d = {"answer": "ERROR %s" % e, "citations": []}
        ans = d.get("answer", "")
        cites = " ".join((c.get("url") or "") for c in d.get("citations", []))
        hits = [bool(re.search(a, ans, re.I)) for a in allx]
        rec = dict(qid=qid, split=split, kind=kind, q=q,
                   strict=all(hits), any=any(hits),
                   abstained=ABSTAIN in ans.lower(),
                   cited=bool(src and re.search(src, cites, re.I)) if src else None,
                   secs=round(time.time() - t, 1), answer=ans[:300])
        rows.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print("%3d/%d %-4s %-5s %s %s" % (
            n, len(qs), qid, split, "OK " if rec["strict"] else
            ("~  " if rec["any"] else "BAD"), q[:52]), flush=True)
    out.close()

    def pc(rs, k):
        return "%3.0f%% (%d/%d)" % (100 * sum(r[k] for r in rs) / len(rs),
                                    sum(r[k] for r in rs), len(rs)) if rs else "-"
    print("\nmode=%s  passages=app's VJLM_TOPK" % mode)
    print("%-22s %-14s %-14s %s" % ("", "strict", "any-pattern", "abstained"))
    groups = [("ALL 98", rows),
              ("v1 tuned (62)", [r for r in rows if r["qid"][0] in "rlkdh"]),
              ("v2 unseen (36)", [r for r in rows if r["qid"].startswith("n")])]
    for name, rs in groups:
        if rs:
            print("%-22s %-14s %-14s %s" % (name, pc(rs, "strict"), pc(rs, "any"),
                                            pc(rs, "abstained")))
    for sp in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == sp]
        print("  split %-14s %-14s %s" % (sp, pc(rs, "strict"), pc(rs, "any")))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
