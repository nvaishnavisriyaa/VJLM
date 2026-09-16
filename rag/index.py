"""Stage 4: build the retrieval index.

Hybrid by design: BM25 (stdlib, always available) + optional dense vectors via
fastembed (ONNX, no torch). If the dense model is unavailable the system still
retrieves - it degrades, it does not break.
"""
import json, math, os, pickle, re, sys
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

def main():
    os.makedirs(IDX, exist_ok=True)
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print(f"loaded {len(rows)} chunks")
    texts = [r["text"] for r in rows]

    print("building BM25 ...", flush=True)
    bm = BM25(texts)
    with open(os.path.join(IDX, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm, "meta": rows}, f)
    print(f"  vocab={len(bm.idf)}  avg_len={bm.avg:.0f}")

    try:
        from fastembed import TextEmbedding
        import numpy as np
        print("building dense vectors (bge-small, ONNX CPU) ...", flush=True)
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        vecs = []
        B = 256
        for i in range(0, len(texts), B):
            vecs.extend(model.embed(texts[i:i + B]))
            if (i // B) % 10 == 0: print(f"  {min(i+B,len(texts))}/{len(texts)}", flush=True)
        V = np.array(list(vecs), dtype="float32")
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        np.save(os.path.join(IDX, "dense.npy"), V)
        print(f"  dense matrix {V.shape}")
    except Exception as e:
        print(f"  dense skipped ({type(e).__name__}: {str(e)[:80]}) - BM25-only retrieval active")

    print("index ready ->", IDX)

if __name__ == "__main__":
    main()
