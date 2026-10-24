"""Screen candidate embedding models without re-embedding the corpus.

For every question the keyword+synonym pool (top POOL chunks, semantic weight 0)
is re-scored by each model, so only ~POOL x questions chunks are embedded per model
instead of 105k. Only models that cost no more than the current one at serving time
are considered: dimension <= 384 and <= ~33M parameters.

    final = base/max(base) + w * minmax(cosine over the pool)

    python evaluation/screen_embedders.py [model ...]
"""
import hashlib, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402
from fastembed import TextEmbedding            # noqa: E402

POOL = 100
WS = [0.0, 0.3, 0.5, 0.8, 1.2, 100.0]          # 100 ~ semantic only
SEARCH_PREFIX = "Represent this sentence for searching relevant passages: "
PREFIX = {                                      # query-side instruction each family expects
    "BAAI/bge-small-en-v1.5": SEARCH_PREFIX,
    "BAAI/bge-small-en": SEARCH_PREFIX,
    "snowflake/snowflake-arctic-embed-xs": SEARCH_PREFIX,
    "snowflake/snowflake-arctic-embed-s": SEARCH_PREFIX,
    "sentence-transformers/all-MiniLM-L6-v2": "",
}
DEFAULT = ["BAAI/bge-small-en-v1.5", "snowflake/snowflake-arctic-embed-xs",
           "snowflake/snowflake-arctic-embed-s", "sentence-transformers/all-MiniLM-L6-v2"]


def hit(c, allx, src):
    if src and not re.search(src, c["url"], re.I):
        return False
    return all(re.search(a, c["text"], re.I) for a in allx)


def main():
    models = sys.argv[1:] or DEFAULT
    r = R.Retriever()
    R.FUSION = (1.0, 0.5, 0.0)
    qs = [(q[0], q[1], q[2], q[3], q[4], q[5]) for q in Q1 + Q2]
    pools = []
    for qid, split, kind, q, allx, src in qs:
        res = r.search(q, role="student", topk=POOL, mode="fusion")
        pools.append(res)
    print("pools: %d questions, %.0f chunks each" % (len(pools), np.mean([len(p) for p in pools])), flush=True)

    def score(ranked_lists):
        n = len(qs)
        ok = [0] * 3
        for (qid, split, kind, q, allx, src), order in zip(qs, ranked_lists):
            rk = next((i + 1 for i, c in enumerate(order[:5]) if hit(c, allx, src)), None)
            if rk:
                ok[0] += 1
            if rk == 1:
                ok[1] += 1
            ok[2] += (1 / rk) if rk else 0
        return "%3.0f%% %3.0f%% %.3f" % (100 * ok[0] / n, 100 * ok[1] / n, ok[2] / n)

    print("%-44s %s" % ("(baseline, no semantic)", score([p[:5] for p in pools])), flush=True)
    for name in models:
        t0 = time.time()
        m = TextEmbedding(name)
        cache = {}
        def emb(texts):
            need = [t for t in dict.fromkeys(texts) if t not in cache]
            for t, v in zip(need, m.embed(need)):
                v = np.asarray(v, dtype="float32")
                cache[t] = v / (np.linalg.norm(v) + 1e-9)
            return np.stack([cache[t] for t in texts])
        pre = PREFIX.get(name, SEARCH_PREFIX)
        per_w = {w: [] for w in WS}
        for (qid, split, kind, q, allx, src), pool in zip(qs, pools):
            D = emb([c["text"] for c in pool])
            qv = np.asarray(list(m.embed([pre + q]))[0], dtype="float32")
            qv /= (np.linalg.norm(qv) + 1e-9)
            cos = D @ qv
            sem = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
            base = np.array([c["score"] for c in pool]); base = base / max(base.max(), 1e-6)
            for w in WS:
                fin = base + w * sem
                per_w[w].append([pool[i] for i in np.argsort(-fin)])
        print("\n%s  (%.0fs, %d chunks embedded)" % (name, time.time() - t0, len(cache)), flush=True)
        for w in WS:
            print("   w=%-6s %s" % ("sem-only" if w == 100.0 else w, score(per_w[w])), flush=True)


if __name__ == "__main__":
    main()
