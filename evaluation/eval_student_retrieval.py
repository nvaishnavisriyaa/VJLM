"""Retrieval-only hit rate on the student question set (RAG-EVAL-002).

No LLM: a question is a HIT when one of the top-k chunks handed to the model
contains the fact (all `all` regexes match inside that single chunk and the
chunk URL matches `src`). Separates "retrieval failed" from "model failed".

  python evaluation/eval_student_retrieval.py                 # coverage, dev+test
  python evaluation/eval_student_retrieval.py --mode bm25
  python evaluation/eval_student_retrieval.py --split dev     # while tuning
  python evaluation/eval_student_retrieval.py --show-miss
The test split is for the final number only; tune against --split dev.
"""
import argparse, os, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402


def is_hit(chunk, allx, src):
    if src and not re.search(src, chunk["url"], re.I):
        return False
    return all(re.search(a, chunk["text"], re.I) for a in allx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="coverage")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test", "hold"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--show-miss", action="store_true")
    ap.add_argument("--scope", default="on", choices=["on", "off"])
    a = ap.parse_args()

    t = time.time()
    r = Retriever()
    print("loaded %d chunks in %.0fs" % (len(r.meta), time.time() - t), flush=True)

    rows, by = [], {}
    for qid, split, kind, q, allx, src in Q:
        if a.split != "all" and split != a.split:
            continue
        chunks = r.search(q, role="student", topk=a.topk, mode=a.mode, scope=(a.scope == "on"))
        rank = next((i + 1 for i, c in enumerate(chunks) if is_hit(c, allx, src)), None)
        rows.append((qid, split, kind, q, rank, chunks))
        by.setdefault(kind, []).append(rank)
        by.setdefault("ALL", []).append(rank)
        by.setdefault("split:" + split, []).append(rank)

    for qid, split, kind, q, rank, chunks in rows:
        print("%-4s %-5s %-10s %s  %s" % (qid, split, kind, ("hit@%d" % rank) if rank else "MISS  ", q[:72]))
        if a.show_miss and rank is None:
            for c in chunks[:5]:
                print("        ->", c["url"][-60:], "|", c["text"][:80].replace("\n", " "))

    print("\nmode=%s topk=%d" % (a.mode, a.topk))
    for k in sorted(by, key=lambda x: (x != "ALL", x)):
        rs = by[k]
        n = len(rs)
        h5 = sum(1 for x in rs if x)
        h1 = sum(1 for x in rs if x == 1)
        mrr = sum(1.0 / x for x in rs if x) / n
        print("%-14s n=%-3d hit@1 %3.0f%%  hit@%d %3.0f%%  MRR %.3f" % (k, n, 100 * h1 / n, a.topk, 100 * h5 / n, mrr))


if __name__ == "__main__":
    main()
