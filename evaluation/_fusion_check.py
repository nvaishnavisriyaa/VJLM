import sys, os, re
sys.path.insert(0, "C:/VJLM"); sys.path.insert(0, "C:/VJLM/evaluation"); os.chdir("C:/VJLM")
from student_questions import Q
from rag import retrieve as R
r = R.Retriever()
def key(c): return (c["url"], c["text"][:80])
same = 0
for qid, split, kind, q, allx, src in Q:
    a = [key(c) for c in r.search(q, topk=5, mode="coverage")]
    b = [key(c) for c in r.search(q, topk=5, mode="fusion")]
    same += a == b
    if a != b: print("DIFF", qid)
print("identical top-5 lists:", same, "of", len(Q))
