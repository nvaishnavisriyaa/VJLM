"""Stage 4: build the retrieval index.

Hybrid by design: BM25 (stdlib, always available) + optional dense vectors via
fastembed (ONNX, no torch). If the dense model is unavailable the system still
retrieves - it degrades, it does not break.
"""
import argparse, json, math, os, pickle, re, sys, time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")
IDX = os.path.join(ROOT, "rag", "index")
STOP = set("a an the of and or to in for on at by is are was were be been with as from this that it its "
           "shall will may can not no if then than which who whom what when where how".split())

def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.tf, self.len = [], []
        df = Counter()
        for d in docs:
            t = tok(d)
            c = Counter(t)
            self.tf.append(c); self.len.append(len(t) or 1)
            df.update(c.keys())
        self.avg = sum(self.len) / max(self.N, 1)
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.inv = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w in c: self.inv[w].append(i)

    def search(self, q, topk=50):
        qt = tok(q); scores = defaultdict(float)
        for w in qt:
            if w not in self.idf: continue
            idf = self.idf[w]
            for i in self.inv[w]:
                f = self.tf[i][w]
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sorted(scores.items(), key=lambda x: -x[1])[:topk]

CACHE = os.path.join(IDX, "dense_cache.npz")
CHECKPOINT_EVERY = 2000


def build_dense(rows, threads=0):
    """Embed only chunks we have never embedded before.

    A chunk_id is a hash of (source_id, page, chunk_index), so an unchanged
    document yields identical ids and its vectors are reused. After the first
    build, a refresh embeds the handful of chunks that actually changed rather
    than all 12,650 - which is what makes an unattended refresh viable on CPU.

    Progress is checkpointed, so a run killed partway resumes instead of
    starting over (this died 4x on the previous machine and lost everything).
    """
    import numpy as np
    from fastembed import TextEmbedding

    cache = {}
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        ids, vecs = z["ids"], z["vecs"]
        cache = {str(k): vecs[i] for i, k in enumerate(ids)}
        print(f"  cache: {len(cache)} vectors already embedded")

    todo = [r for r in rows if r["chunk_id"] not in cache]
    print(f"  to embed: {len(todo)} of {len(rows)}")

    if todo:
        # onnxruntime otherwise picks its own thread count; on a many-core CPU
        # the default leaves most of the machine idle.
        kw = {"threads": threads} if threads else {}
        model = TextEmbedding("BAAI/bge-small-en-v1.5", **kw)
        done, t0 = 0, time.time()
        B = 128
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            for r, v in zip(batch, model.embed([r["text"] for r in batch])):
                cache[r["chunk_id"]] = np.asarray(v, dtype="float32")
            done += len(batch)
            if done % CHECKPOINT_EVERY < B or done == len(todo):
                save_cache(cache, np)
                rate = done / max(time.time() - t0, 1e-6)
                left = (len(todo) - done) / max(rate, 1e-6) / 60
                print(f"  {done}/{len(todo)}  {rate:.0f}/s  ~{left:.1f} min left", flush=True)

    # Row order must match bm25 meta order, so build the matrix from rows.
    V = np.stack([cache[r["chunk_id"]] for r in rows]).astype("float32")
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    np.save(os.path.join(IDX, "dense.npy"), V)
    return V.shape


def save_cache(cache, np):
    tmp = CACHE + ".tmp.npz"
    np.savez(tmp, ids=np.array(list(cache), dtype=object).astype(str),
             vecs=np.stack(list(cache.values())).astype("float32"))
    os.replace(tmp, CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-only", action="store_true",
                    help="skip dense; BM25 alone is enough to serve")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 0)
    a = ap.parse_args()

    os.makedirs(IDX, exist_ok=True)
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print(f"loaded {len(rows)} chunks")
    texts = [r["text"] for r in rows]

    print("building BM25 ...", flush=True)
    t0 = time.time()
    bm = BM25(texts)
    with open(os.path.join(IDX, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm, "meta": rows}, f)
    print(f"  vocab={len(bm.idf)}  avg_len={bm.avg:.0f}  {time.time()-t0:.1f}s")

    if a.bm25_only:
        print("index ready (BM25 only) ->", IDX)
        return

    try:
        print("building dense vectors (bge-small, ONNX CPU) ...", flush=True)
        shape = build_dense(rows, a.threads)
        print(f"  dense matrix {shape}")
    except Exception as e:
        print(f"  dense skipped ({type(e).__name__}: {str(e)[:80]}) - BM25-only retrieval active")

    print("index ready ->", IDX)

if __name__ == "__main__":
    main()
