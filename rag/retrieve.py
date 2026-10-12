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


IDF = {}   # set by Retriever; used to focus passages on the question

ENRICHED = os.path.join(ROOT, "preprocessing", "out", "rag_chunks_enriched.jsonl")
SYNC_STATE = os.path.join(ROOT, "ingestion", "state", "sync_state.json")


class Retriever:
    def __init__(self):
        with open(os.path.join(IDX, "bm25.pkl"), "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        global IDF
        IDF = self.bm25.idf
        self.meta = blob["meta"]
        # Query expansion (RAG-SYN-001). Gated on the indexed vocabulary so a
        # WordNet synonym that appears in no VNRVJIET document is discarded
        # before it can add noise. Default OFF - enable per call, or set
        # VJLM_EXPAND=1, once the benchmark says it helps.
        from rag.synonyms import Expander
        self.expander = Expander(
            vocab=set(self.bm25.idf),
            weight=float(os.environ.get("VJLM_EXPAND_WEIGHT", "0.35")),
            max_per_term=int(os.environ.get("VJLM_EXPAND_MAX", "3")))
        self.expand_default = os.environ.get("VJLM_EXPAND", "0") == "1"
        # Synonym-aware coverage (RAG-SYN-002): 1.0 literal / 0.5 synonym / 0
        # none, IDF-weighted. Measures BREADTH of query coverage, which BM25
        # (a term-frequency measure) does not - see rag/coverage.py.
        from rag.coverage import CoverageScorer
        self.coverage = CoverageScorer(
            self.bm25, self.expander,
            w_syn=float(os.environ.get("VJLM_COVERAGE_SYN", "0.5")))
        self.w_coverage = float(os.environ.get("VJLM_W_COVERAGE", "0.0"))
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
        # Publication date per source, from the crawl's Last-Modified header
        # (ingestion/crawler/fetch.py: http_date). Used to settle sources that
        # contradict each other, and shown with the citation.
        self.doc_date = {}
        try:
            with open(SYNC_STATE, encoding="utf-8") as f:
                for sid, v in json.load(f).items():
                    if v.get("last_modified"):
                        self.doc_date[sid] = v["last_modified"]
        except (OSError, ValueError):
            pass
        for r in self.meta:
            d = self.doc_date.get(r.get("source_id"))
            if d:
                r["published"] = d

        # Per-chunk document profile for query scoping (RAG-SCOPE-001).
        from rag import scope as _scope
        self._scope = _scope
        _cache = {}
        self.profile = []
        for r in self.meta:
            k = (r.get("url"), r.get("regulation"), r.get("category"), r.get("document_type"))
            if k not in _cache:
                _cache[k] = _scope.doc_profile(r)
            self.profile.append(_cache[k])
        self.scope_default = os.environ.get("VJLM_SCOPE", "1") == "1"
        self.dense = None
        self.embed = None
        dpath = os.path.join(IDX, "dense.npy")
        if os.path.exists(dpath):
            try:
                import numpy as np
                from fastembed import TextEmbedding
                self.np = np
                self.dense = np.load(dpath)
                # Row i of dense.npy must be chunk i of the BM25 index. If a
                # dense build failed after BM25 was rebuilt, the matrix is from
                # an older chunk list and every dense hit returns the WRONG
                # chunk's text - silently, with a plausible-looking citation.
                # That happened on 2026-09-30 (13,051 rows vs 12,978 chunks).
                # A missing signal is recoverable; a wrong one is not.
                if self.dense.shape[0] != len(self.meta):
                    print("[retrieve] dense.npy has %d rows but the index has %d "
                          "chunks - dense DISABLED until rag/index.py rebuilds it"
                          % (self.dense.shape[0], len(self.meta)), flush=True)
                    self.dense = None
                else:
                    self.embed = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:
                self.dense = None

    def _allowed(self, row, role):
        vis = ROLE_VISIBILITY.get(role, {"PUBLIC"})
        if row.get("visibility", "PUBLIC") not in vis:
            return False
        aud = row.get("audience", "student|faculty").split("|")
        return role in aud or "guest" == role

    def search(self, query, role="student", topk=6, pool=60, mode="hybrid",
               expand=None, scope=None):
        """mode: 'bm25' | 'dense' | 'hybrid'. Falls back to bm25 if dense is absent.

        expand: WordNet query expansion on the lexical arm. None = the default
        from VJLM_EXPAND. Only the lexical arm is expanded; the dense arm
        already generalises over wording, so expanding it would be redundant
        at best and would blur the query vector at worst.
        """
        if self.dense is None and mode in ("dense", "hybrid"):
            mode = "bm25"
        # Each mode is a weighting over three signals:
        #   lex = BM25 (term frequency), cov = coverage (breadth),
        #   dns = cosine over embeddings (meaning).
        WEIGHTS = {
            "bm25":     (1.0, 0.0, 0.0),
            "dense":    (0.0, 0.0, 1.0),
            "coverage": (0.0, 1.0, 0.0),
            "lexical":  (0.6, 0.4, 0.0),   # BM25 + coverage, no embeddings
            "hybrid":   (0.5, 0.0, 0.5),
            "hybrid3":  (0.35, 0.25, 0.40),
        }
        if mode not in WEIGHTS:
            raise ValueError("unknown mode %r; known: %s"
                             % (mode, ", ".join(sorted(WEIGHTS))))
        w_lex, w_cov, w_dns = WEIGHTS[mode]
        if self.w_coverage and mode in ("bm25", "hybrid"):
            w_cov = self.w_coverage          # env override for A/B runs
        scores = defaultdict(float)

        # --- lexical ---
        if expand is None:
            expand = self.expand_default
        self.last_expansion = None
        if w_lex:
            from rag.index import tok as _tok
            terms = [(w, 1.0) for w in _tok(query)]
            if expand:
                terms = self.expander.expand([w for w, _ in terms])
                self.last_expansion = self.expander.explain()
            lex = self.bm25.search_weighted(terms, topk=pool)
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

        # --- coverage (breadth) over the pool the other arms surfaced ---
        # Scored on candidates only: coverage is a re-ranking signal, not a
        # retriever. Running it over all 13k chunks would cost far more and
        # add nothing, since a document no arm surfaced has no query term in
        # it and would score ~0 anyway.
        if scope is None:
            scope = self.scope_default
        qi = self._scope.parse_query(query) if scope else None
        if w_cov:
            from rag.index import tok as _t
            toks = _t(query)
            pool_ids = list(scores.keys()) if scores else []
            if scope:
                # BM25 only surfaces what its own scoring likes; on a corpus
                # that is 62% syllabus text the right chunk is often rank 200+.
                # Take a wide pool, plus the best of each family so a curated
                # site page or the current rulebook is always a candidate.
                allsc = self.bm25.search_weighted([(w, 1.0) for w in toks], topk=100000)
                seen_ids = set(pool_ids)
                per_fam = defaultdict(int)
                for n, (i, _s) in enumerate(allsc):
                    fam = self.profile[i][0]
                    if n < 300 or per_fam[fam] < 120:
                        per_fam[fam] += 1
                        if i not in seen_ids:
                            seen_ids.add(i); pool_ids.append(i)
            elif not pool_ids:
                pool_ids = [i for i, _ in self.bm25.search(query, topk=pool)]
            bm_norm = {}
            if scope:
                mx_b = allsc[0][1] if allsc else 1.0
                bm_norm = {i: v / mx_b for i, v in allsc[:2000]}
            terms = self.coverage.build_terms(self._scope.content_terms(toks) if scope else toks)
            terms = [(t, idf, syns, self._scope.variants(t) if scope else ()) for t, idf, syns in terms]
            den = sum(idf for _, idf, _, _ in terms) or 1.0
            ids = (self._scope.id_weights(self._scope.identifiers(query), self.bm25.idf)
                   if scope else ())
            for i in pool_ids:
                cv = self._scope.cover_score(self.coverage.tf[i], terms, self.coverage.w_syn)
                if scope and cv and self._scope.is_toc(self.meta[i]["text"]):
                    cv *= self._scope.W["toc"]
                if scope and cv:
                    # a chunk whose opening words (section heading) match the
                    # question is more likely the section that answers it
                    head = set(_t(self.meta[i]["text"][:160]))
                    cv += self._scope.W["head"] * sum(idf for t, idf, _, var in terms if t in head or any(x in head for x in var)) / den
                if scope and cv and ids:
                    # the "O" in "O grade", the SEE in "CIE and SEE": dropped by
                    # the tokenizer, recovered here from the raw text with case
                    cv += self._scope.W["ident"] * self._scope.id_bonus(self.meta[i]["text"], ids, den)
                scores[i] += w_cov * cv
                if scope and bm_norm:
                    scores[i] += self._scope.W["bm25_tie"] * bm_norm.get(i, 0.0)

        # --- RBAC filter BEFORE ranking is returned ---
        boosts = ROLE_BOOST.get(role, {})
        ranked = []
        for i, s in scores.items():
            row = self.meta[i]
            if not self._allowed(row, role):
                continue
            m = boosts.get(row.get("category", ""), 1.0)
            if scope:
                m *= self._scope.factor(self.profile[i], qi)
            ranked.append((s * m, i))
        ranked.sort(reverse=True)
        fam_cap = self._scope.quota(qi, topk) if scope else {}
        fam_n = defaultdict(int)
        src_n = defaultdict(int)

        # one chunk per document keeps citations diverse; duplicates dropped
        out = []
        for s, i in ranked:
            row = self.meta[i]
            if row.get("duplicate_of"):
                continue
            fam = self.profile[i][0]
            per_doc = self._scope.W["per_doc"].get(fam, 1) if scope else 1
            if src_n[row["source_id"]] >= per_doc:
                continue
            if fam_cap and fam_n[fam] >= fam_cap.get(fam, topk):
                continue
            fam_n[fam] += 1
            src_n[row["source_id"]] += 1
            out.append(dict(row, score=round(s, 4)))
            if len(out) >= topk:
                break
        return out


SYSTEM = ("You are VJLM, the Vignana Jyothi Language Model - the AI assistant for "
          "VNR Vignana Jyothi Institute of Engineering & Technology (VNRVJIET). "
          "Answer ONLY from the CONTEXT provided. Cite sources as [1], [2]. "
          "If the context does not contain the answer, say: 'I could not find this "
          "in the available VNRVJIET sources.' Never invent dates, names, fees or policies.")

# The wording "The user is a STUDENT. Explain plainly and practically." was the
# exact cue in the role-differentiation training rows whose answer template is
# "In simple terms: ... check this against your own regulation batch". Keeping
# that cue made every student answer end with an unsupported hedge, so the
# audience note is phrased differently (APP-002).
ROLE_NOTE = {
    "student": "Audience: a current VNRVJIET student. Be clear and direct.",
    "faculty": "Audience: a VNRVJIET faculty member. Be precise and procedural.",
    "guest":   "Audience: a visitor. Answer from public information only.",
}


from rag.focus import focus as _focus  # noqa: E402


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
            i + 1, c["url"], c["page"], meta, redact(_focus(query, c["text"], IDF))))
    ctx = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM + "\n" + ROLE_NOTE.get(role, "")},
        {"role": "user", "content": "CONTEXT:\n{}\n\nQUESTION: {}".format(ctx, query)},
    ]
