import sys, os, re
sys.path.insert(0, "C:/VJLM"); sys.path.insert(0, "C:/VJLM/evaluation"); os.chdir("C:/VJLM")
from student_questions import Q
from rag import retrieve as R
r = R.Retriever()
def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I): return False
    return all(re.search(a, c["text"], re.I) for a in allx)
def ranks(w):
    R.FUSION = w; out = {}
    for qid, split, kind, q, allx, src in Q:
        res = r.search(q, topk=5, mode="fusion")
        out[qid] = next((i + 1 for i, c in enumerate(res) if hit(c, allx, src)), None)
    return out
A, B = ranks((1, .5, 0)), ranks((1, 1, .5))
for qid, split, kind, q, *_ in Q:
    if A[qid] != B[qid]:
        print("%-4s %-5s %-10s sem0:%-4s sem.5:%-4s %s" % (qid, split, kind, A[qid], B[qid], q[:70]))
