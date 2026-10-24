"""Confirm every question's answer pattern exists in the corpus.

A question whose pattern matches no chunk can never be a hit, and would make
every retriever look worse for no reason. Prints the count of matching chunks and
a snippet around the first match.

    python evaluation/check_questions.py
"""
import json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from student_questions_v2 import Q2  # noqa: E402

rows = [json.loads(l) for l in open(os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl"),
                                    encoding="utf-8")]
bad = 0
for qid, split, kind, q, allx, src in Q2:
    hits = [r for r in rows if re.search(src, r["url"], re.I)
            and all(re.search(a, r["text"], re.I) for a in allx)]
    flag = "OK " if hits else "NONE"
    bad += not hits
    snip = ""
    if hits:
        m = re.search(allx[0], hits[0]["text"], re.I)
        snip = re.sub(r"\s+", " ", hits[0]["text"][max(0, m.start() - 40): m.end() + 60])
    print("%s %-4s %-5s n=%-3d %s" % (flag, qid, split, len(hits), snip.encode("ascii", "replace").decode()[:110]))
print("\n%d of %d patterns match nothing" % (bad, len(Q2)))
