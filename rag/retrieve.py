"""Role-aware hybrid retrieval.

Order matters: RBAC/visibility filtering happens BEFORE ranking, so an
unauthorised chunk can never reach the prompt builder. The model is never
the security boundary.
"""
import json, os, pickle, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "rag", "index")
sys.path.insert(0, ROOT)

# index.py builds the pickle while running as __main__, so the BM25 class is
# recorded as __main__.BM25. Make that name resolvable when loading elsewhere.
from rag.index import BM25 as _BM25  # noqa: E402
sys.modules["__main__"].BM25 = _BM25

# What each role is allowed to see. Phase 1 is public-only, but the gate is
# real code from day one so login-only data can be added without redesign.
ROLE_VISIBILITY = {
    "student": {"PUBLIC"},
    "faculty": {"PUBLIC"},
    "guest":   {"PUBLIC"},
}
# Same public source, different ranking emphasis per role.
ROLE_BOOST = {
    "student": {"academic_regulations": 1.15, "examinations": 1.15, "syllabus": 1.10,
                "fees_scholarships": 1.10, "clubs_events": 1.05, "facilities": 1.05},
    "faculty": {"departments": 1.15, "research": 1.15, "governance_grievance": 1.10,
                "academic_regulations": 1.10, "placements": 1.05},
}


ENRICHED = os.path.join(ROOT, "preprocessing", "out", "rag_chunks_enriched.jsonl")


class Retriever:
    def __init__(self):
        with open(os.path.join(IDX, "bm25.pkl"), "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        self.meta = blob["meta"]
        # Merge Layer-1 enrichment (department, academic_year, PII flag,
        # duplicate marker) by chunk_id. Metadata only - embeddings untouched.
        if os.path.exists(ENRICHED):
            extra = {}
            with open(ENRICHED, encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    extra[e["chunk_id"]] = e
            merged = 0
            for r in self.meta:
                e = extra.get(r["chunk_id"])
                if e:
                    for k in ("department", "academic_year", "regulation",
                              "has_pii", "duplicate_of"):
                        if k in e:
                            r[k] = e[k]
                    merged += 1
            self.enriched = merged
        self.dense = None
        self.embed = None
        dpath = os.path.join(IDX, "dense.npy")
        if os.path.exists(dpath):
            try:
                import numpy as np
                from fastembed import TextEmbedding
                self.np = np
                self.dense = np.load(dpath)
                self.embed = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:
                self.dense = None

    def _allowed(self, row, role):
        vis = ROLE_VISIBILITY.get(role, {"PUBLIC"})
        if row.get("visibility", "PUBLIC") not in vis:
            return False
        aud = row.get("audience", "student|faculty").split("|")
        return role in aud or "guest" == role

    def search(self, query, role="student", topk=6, pool=60, mode="hybrid"):
        """mode: 'bm25' | 'dense' | 'hybrid'. Falls back to bm25 if dense is absent."""
        if self.dense is None:
            mode = "bm25"
        w_lex = {"bm25": 1.0, "dense": 0.0, "hybrid": 0.5}[mode]
        w_dns = {"bm25": 0.0, "dense": 1.0, "hybrid": 0.5}[mode]
        scores = defaultdict(float)

        # --- lexical ---
        if w_lex:
            lex = self.bm25.search(query, topk=pool)
            if lex:
                mx = max(s for _, s in lex) or 1.0
                for i, s in lex:
                    scores[i] += w_lex * (s / mx)

        # --- dense (if available) ---
        if w_dns and self.dense is not None and self.embed is not None:
            qv = list(self.embed.embed([query]))[0]
            qv = self.np.asarray(qv, dtype="float32")
            qv /= (self.np.linalg.norm(qv) + 1e-9)
            sims = self.dense @ qv
            top = self.np.argpartition(-sims, min(pool, len(sims) - 1))[:pool]
            mx = float(sims[top].max()) or 1.0
            for i in top:
                scores[int(i)] += w_dns * float(sims[i]) / mx

        # --- RBAC filter BEFORE ranking is returned ---
        boosts = ROLE_BOOST.get(role, {})
        ranked = []
        for i, s in scores.items():
            row = self.meta[i]
            if not self._allowed(row, role):
                continue
            ranked.append((s * boosts.get(row.get("category", ""), 1.0), i))
        ranked.sort(reverse=True)

        # one chunk per document keeps citations diverse; duplicates dropped
        seen, out = set(), []
        for s, i in ranked:
            row = self.meta[i]
            if row.get("duplicate_of"):
                continue
            if row["source_id"] in seen:
                continue
            seen.add(row["source_id"])
            out.append(dict(row, score=round(s, 4)))
            if len(out) >= topk:
                break
        return out


SYSTEM = ("You are VJLM, the Vignana Jyothi Language Model - the AI assistant for "
          "VNR Vignana Jyothi Institute of Engineering & Technology (VNRVJIET). "
          "Answer ONLY from the CONTEXT provided. Cite sources as [1], [2]. "
          "If the context does not contain the answer, say: 'I could not find this "
          "in the available VNRVJIET sources.' Never invent dates, names, fees or policies.")

ROLE_NOTE = {
    "student": "The user is a STUDENT. Explain plainly and practically.",
    "faculty": "The user is FACULTY. Be precise and procedural.",
    "guest":   "The user is a visitor. Answer from public information only.",
}


def build_prompt(query, chunks, role="student"):
    """Assembly order: system -> role -> context -> question -> output rules.
    Access control is NOT in the prompt; it already happened in retrieval.
    PII is redacted here so the model never sees emails or phone numbers."""
    from preprocessing.enrich import redact
    parts = []
    for i, c in enumerate(chunks):
        tag = []
        if c.get("department"):
            tag.append(c["department"])
        if c.get("regulation"):
            tag.append(c["regulation"])
        if c.get("academic_year"):
            tag.append(c["academic_year"])
        meta = (" [" + ", ".join(tag) + "]") if tag else ""
        parts.append("[{}] {} (page {}){}\n{}".format(
            i + 1, c["url"], c["page"], meta, redact(c["text"][:900])))
    ctx = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM + "\n" + ROLE_NOTE.get(role, "")},
        {"role": "user", "content": "CONTEXT:\n{}\n\nQUESTION: {}".format(ctx, query)},
    ]
