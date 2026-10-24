import sys, os, re
sys.path.insert(0, "C:/VJLM"); sys.path.insert(0, "C:/VJLM/evaluation"); os.chdir("C:/VJLM")
from student_questions_v2 import Q2
from rag import retrieve as R
R.FUSION = (1.0, 0.5, 0.8)
r = R.Retriever()
def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I): return False
    return all(re.search(a, c["text"], re.I) for a in allx)
for qid, split, kind, q, allx, src in Q2:
    res = r.search(q, topk=5, mode="fusion")
    rk = next((i + 1 for i, c in enumerate(res) if hit(c, allx, src)), None)
    if rk != 1:
        print("\n%s rank=%s  %s" % (qid, rk, q))
        for c in res[:3]:
            print("    ", c["url"][-48:], "|", re.sub(r"\s+", " ", c["text"][:70]).encode("ascii", "replace").decode())
