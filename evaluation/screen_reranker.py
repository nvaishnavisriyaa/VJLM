"""Does a local cross-encoder reranker convert hit@20 into hit@5?

Retrieval leaves 92% of answers inside the top 20 but only 83% inside the top 5
(evaluation/depth_curve.py), so the last gap is ordering. A cross-encoder reads
the question and a passage TOGETHER, which is what the three cheap signals cannot
do. It runs on CPU over a handful of candidates, never over the corpus.

Reports hit@5 before and after reranking, and the per-query cost, so the latency
is a measurement and not a guess.

    python evaluation/screen_reranker.py [depth] [model]
"""
import os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def rate(lists, qs, k=5):
    n = len(qs)
    h = sum(1 for (q, lst) in zip(qs, lists)
            if any(hit(c, q[4], q[5]) for c in lst[:k]))
    h1 = sum(1 for (q, lst) in zip(qs, lists) if lst and hit(lst[0], q[4], q[5]))
    return "hit@5 %3.0f%%  hit@1 %3.0f%%" % (100 * h / n, 100 * h1 / n)


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    name = sys.argv[2] if len(sys.argv) > 2 else "Xenova/ms-marco-MiniLM-L-6-v2"
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    R.FUSION = (1.0, 0.5, 1.5)
    r = R.Retriever()
    qs = Q1 + Q2
    pools = [r.search(q[3], role="student", topk=depth, mode="fusion") for q in qs]
    print("depth=%d  model=%s" % (depth, name), flush=True)
    print("before      %s" % rate(pools, qs), flush=True)

    ce = TextCrossEncoder(model_name=name)
    t0 = time.time()
    reranked, blended = [], []
    for q, pool in zip(qs, pools):
        texts = [(c.get("context", "") + "\n" + c["text"]).strip() for c in pool]
        sc = list(ce.rerank(q[3], texts)) if texts else []
        order = sorted(range(len(pool)), key=lambda i: -sc[i])
        reranked.append([pool[i] for i in order])
        # blended: cross-encoder rank combined with the retriever's own order,
        # so a confident retrieval is not overturned by a single model's opinion
        lo, hi = (min(sc), max(sc)) if sc else (0, 1)
        span = max(hi - lo, 1e-6)
        mx = max((c["score"] for c in pool), default=1.0) or 1.0
        bl = sorted(range(len(pool)),
                    key=lambda i: -((sc[i] - lo) / span + pool[i]["score"] / mx))
        blended.append([pool[i] for i in bl])
    dt = time.time() - t0
    print("reranked    %s" % rate(reranked, qs), flush=True)
    print("blended     %s" % rate(blended, qs), flush=True)
    print("cost        %.0f ms per query (%d candidates)" % (1000 * dt / len(qs), depth))


if __name__ == "__main__":
    main()
