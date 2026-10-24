"""Role-aware hybrid retrieval.

Order matters: RBAC/visibility filtering happens BEFORE ranking, so an
unauthorised chunk can never reach the prompt builder. The model is never
the security boundary.
"""
import json, os, pickle, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.environ.get("VJLM_INDEX_DIR") or os.path.join(ROOT, "rag", "index")
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

ENRICHED = os.path.join(os.environ.get("VJLM_OUT_DIR") or os.path.join(ROOT, "preprocessing", "out"), "rag_chunks_enriched.jsonl")
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
            # The idf MAP, not just its keys: `w in vocab` behaves identically on
            # a dict, and the IDF-gated WordNet filter needs the values to ask
            # whether a synonym is as discriminating as the term it replaces.
            vocab=self.bm25.idf,
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
        self._glossary = None
        self._hubs = None
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

    def _hub_index(self):
        """-> {url: 0-1} for pages that are the PARENT of several other pages.

        A page with many children is a section overview, and a question about
        the set ("which clubs can I join", "how many branches are there") is
        answered there, not on a leaf. Measured cause, from
        evaluation/miss_report.py: forty individual club pages each match
        "club" and bury /campus-life/clubs. The parent/child shape is already
        in the crawled URLs, so nothing has to be listed by hand - and the
        bonus only applies to a page that matched the query at all.
        """
        if self._hubs is None:
            import math
            kids = defaultdict(set)
            for u in {m.get("url") or "" for m in self.meta}:
                p = u.split("#")[0].rstrip("/")
                while p.count("/") > 3:          # stop above the bare domain
                    p = p.rsplit("/", 1)[0]
                    kids[p].add(u)
            self._hubs = {u: min(1.0, math.log1p(len(c)) / math.log(11.0))
                          for u, c in kids.items() if len(c) >= 3}
        return self._hubs

    def search(self, query, role="student", topk=6, pool=60, mode="coverage",
               expand=None, scope=None):
        """mode: 'coverage' (default, the measured best) | 'bm25' | 'dense' | 'hybrid' ...
        Dense-based modes fall back to coverage if the dense index is absent.

        expand: WordNet query expansion on the lexical arm. None = the default
        from VJLM_EXPAND. Only the lexical arm is expanded; the dense arm
        already generalises over wording, so expanding it would be redundant
        at best and would blur the query vector at worst.
        """
        if self.dense is None and mode in ("dense", "hybrid", "hybrid3"):
            mode = "coverage"
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
            "fusion":   (0.0, 1.0, 0.0),   # three graded signals; weights in FUSION
        }
        if mode not in WEIGHTS:
            raise ValueError("unknown mode %r; known: %s"
                             % (mode, ", ".join(sorted(WEIGHTS))))
        w_lex, w_cov, w_dns = WEIGHTS[mode]
        fusion = mode == "fusion" and self.dense is not None   # no embeddings: coverage alone
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
            sims = None
            if fusion and self.embed is not None:
                # The semantic arm RETRIEVES here, it does not only re-rank.
                # Measured (evaluation/dense_recall.py + diagnose_misses.py): 14 of
                # 25 misses were chunks BM25 ranks 300-9000th, which no weighting
                # over a BM25-only pool can ever reach - while cosine puts them at
                # 2, 16, 22, 37. The reverse also happens ("leftover food at the
                # hostels": BM25 7, cosine 908), so neither arm can own the pool.
                # The candidates are therefore the UNION of both rankings and every
                # candidate is scored by all three signals.
                np = self.np
                qv = np.asarray(list(self.embed.embed([FUSION_QPREFIX + query]))[0], dtype="float32")
                qv /= (np.linalg.norm(qv) + 1e-9)
                sims = self.dense @ qv
                have = set(pool_ids)
                k = min(FUSION_DENSE_POOL, len(sims) - 1)
                top = np.argpartition(-sims, k)[:k]
                for i in top[np.argsort(-sims[top])]:
                    if int(i) not in have:
                        have.add(int(i)); pool_ids.append(int(i))
            arms = {}
            bm_norm = {}
            if scope:
                mx_b = allsc[0][1] if allsc else 1.0
                bm_norm = {i: v / mx_b for i, v in allsc[:2000]}
            terms = self.coverage.build_terms(self._scope.content_terms(toks) if scope else toks)
            # IDF is raised to a power before coverage uses it. Coverage is an
            # IDF-weighted MEAN, so it rewards BREADTH: a page covering three
            # common words of "in which area of Hyderabad is the campus" scored
            # 1.00 while the page holding the answer scored 0.64, because the
            # answer is a place name the question cannot contain. Measured per
            # arm with evaluation/why_miss.py, that shape was 6 of 12 reachable
            # misses. The power makes the question's DISCRIMINATING term
            # dominate instead of its topic words; it is still the same
            # synonym-aware coverage, just concentrated.
            terms = [(t, idf ** IDF_POW, syns, self._scope.variants(t) if scope else ())
                     for t, idf, syns in terms]
            den = sum(idf for _, idf, _, _ in terms) or 1.0
            ids = (self._scope.id_weights(self._scope.identifiers(query), self.bm25.idf)
                   if scope else ())
            url_w = [(t, idf) for t, idf, _, _ in terms]
            all_ids = self._scope.identifiers(query) if scope and qi["define"] else ()
            if all_ids:
                # an abbreviation table mentions each term once, so BM25 ranks it
                # far below the chunks that use the term often: make them candidates
                if self._glossary is None:
                    self._glossary = [n for n, m in enumerate(self.meta)
                                      if self._scope.is_glossary(m["text"])]
                have = set(pool_ids)
                pool_ids += [n for n in self._glossary if n not in have]
            # Semantic scores are min-maxed over the CANDIDATES, not the corpus, so
            # the arm spans 0-1 like coverage does and the three weights are
            # comparable. (A corpus-wide percentile left every candidate near 1.0
            # on a query whose answer is everywhere, and near 0 on a rare one.)
            dn = {}
            if sims is not None and pool_ids:
                sel = self.np.array(pool_ids)
                sv = sims[sel]
                lo, hi = float(sv.min()), float(sv.max())
                span = max(hi - lo, 1e-6)
                dn = {int(i): (float(sims[i]) - lo) / span for i in pool_ids}
            for i in pool_ids:
                cv = self._scope.cover_score(self.coverage.tf[i], terms, self.coverage.w_syn)
                tm, cv_base = 1.0, cv
                if scope and cv and self._scope.is_toc(self.meta[i]["text"]):
                    cv *= self._scope.W["toc"]
                    tm = self._scope.W["toc"]
                cv_pre = cv
                if scope and cv:
                    # a chunk whose opening words (section heading) match the
                    # question is more likely the section that answers it
                    head = set(_t((self.meta[i].get("context", "") + " " + self.meta[i]["text"])[:260]))
                    cv += self._scope.W["head"] * sum(idf for t, idf, _, var in terms if t in head or any(x in head for x in var)) / den
                if scope and cv and ids:
                    # the "O" in "O grade", the SEE in "CIE and SEE": dropped by
                    # the tokenizer, recovered here from the raw text with case
                    cv += self._scope.W["ident"] * self._scope.id_bonus(self.meta[i].get("context", "") + " " + self.meta[i]["text"], ids, den)
                if scope and cv and url_w:
                    # the branch/programme often appears only in the file name
                    # (II-B.Tech-CSBS-R25.pdf): its tables never repeat it
                    ut = set(re.findall(r"[a-z0-9]+", self.meta[i].get("url", "").lower()))
                    cv += self._scope.W["url"] * sum(w for t, w in url_w if t in ut) / den
                if scope and cv and self._scope.W["hub"]:
                    hb = self._hub_index().get(self.meta[i].get("url") or "", 0.0)
                    if hb:
                        cv += self._scope.W["hub"] * hb * cv_base
                if scope and cv and qi["listy"] and self._scope.W["enum"]:
                    cv += self._scope.W["enum"] * self._scope.enum_bonus(
                        self.coverage.tf[i], terms, den)
                if scope and qi["define"] and all_ids:
                    cv += self._scope.W["glossary"] * self._scope.glossary_bonus(self.meta[i]["text"], all_ids)
                if fusion:
                    f_kw, f_syn, f_sem = FUSION
                    ex, sy = self._scope.cover_parts(self.coverage.tf[i], terms)
                    # heading / identifier / file-name bonuses are keyword evidence
                    kw = ex * tm + (cv - cv_pre)
                    scores[i] += f_kw * kw + f_syn * sy * tm + f_sem * dn.get(i, 0.0)
                    if scope and bm_norm:
                        scores[i] += self._scope.W["bm25_tie"] * bm_norm.get(i, 0.0)
                    if RRF_K:
                        arms[i] = (kw, sy * tm, dn.get(i, 0.0))
                    if PARTS is not None:
                        PARTS[i] = dict(kw=f_kw * kw, syn=f_syn * sy * tm,
                                        sem=f_sem * dn.get(i, 0.0), raw_kw=ex,
                                        bonus=cv - cv_pre, sem_raw=dn.get(i, 0.0))
                    continue
                scores[i] += w_cov * cv
                if scope and bm_norm:
                    scores[i] += self._scope.W["bm25_tie"] * bm_norm.get(i, 0.0)

            # Reciprocal-rank fusion, as an alternative to the weighted sum.
            # The three arms are on different scales per query (two are shares of
            # the query's IDF mass, one is a min-maxed cosine), so a weighted sum
            # lets whichever arm happens to spread widest dominate. RRF combines
            # RANKS instead, which is scale-free and is the usual way a hybrid is
            # mixed. Same pool, same bonuses: only the combination changes.
            if RRF_K and arms:
                f_kw, f_syn, f_sem = FUSION
                order = {}
                for a, (w, name) in enumerate(((f_kw, "kw"), (f_syn, "syn"), (f_sem, "sem"))):
                    if not w:
                        continue
                    rank = sorted(arms, key=lambda i: -arms[i][a])
                    for r, i in enumerate(rank):
                        if arms[i][a] > 0:
                            order[i] = order.get(i, 0.0) + w / (RRF_K + r + 1)
                for i in arms:
                    scores[i] = order.get(i, 0.0)
                    if scope and bm_norm:
                        scores[i] += 1e-6 * bm_norm.get(i, 0.0)

            # Sentence-level pass, over the leaders only. The three arms all read
            # the whole chunk, so none of them can tell "one sentence answers
            # this" from "the page mentions these words somewhere". Restricted to
            # the top candidates it is a reranking signal that costs a few ms and
            # no memory - see rag/scope.py:best_unit.
            if SENT_W and terms:
                lead = sorted(scores, key=lambda i: -scores[i])[:SENT_N]
                for i in lead:
                    m = self.meta[i]
                    scores[i] += SENT_W * self._scope.best_unit(
                        (m.get("context", "") + "\n" + m["text"]), terms, den,
                        self.coverage.w_syn)

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

        # A few chunks per document keeps the five citations in a prompt diverse.
        # The caps scale with topk: they are a presentation rule for a 5-passage
        # prompt, and left fixed they became an absolute recall ceiling - with 3
        # chunks allowed per rulebook, a clause ranked 4th within the R25 PDF
        # could not be returned at ANY depth (measured: hit@100 stuck at 89%).
        grow = max(1, topk // 5) * PER_DOC_SCALE
        # One document may fill every slot. The old fixed caps (3 chunks of a
        # rulebook, 1 of a syllabus PDF) were a presentation rule for a 5-passage
        # prompt, and they cost real accuracy: when the answer IS in the R25
        # rulebook the best five passages are often all from it, and refusing the
        # 4th in the name of citation diversity threw the answer away. Measured on
        # 98 questions, lifting the caps moved hit@5 79% -> 83% and hit@100 89% ->
        # 95%; even reserving ONE slot for a second source cost 2 points, so no
        # slot is reserved. Diversity is still served by the duplicate filter and
        # by ranking, not by discarding the best passage.
        # ... but that measurement was retrieval-only, and the end-to-end test
        # then dropped 24/26 -> 22/26 for a reason hit@5 cannot see: with no cap,
        # ONE document monopolises the prompt. "Where can I study quietly and
        # borrow books" returned four passages of Adminstrative Manual.pdf and
        # never showed the library page; the credits question showed four
        # branches' syllabus tables and the model answered from the wrong branch.
        # So the two jobs are separated. Retrieval DEPTH stays uncapped, which is
        # what the recall ceiling needed, while the passages a PROMPT shows keep
        # a share-based cap that only binds when topk is small.
        doc_cap = max(DOC_CAP_MIN, int(topk * DOC_CAP_FRAC))
        # A rulebook question is answered by the rulebook, several clauses of it,
        # and capping that document is how the answer gets thrown away. Measured
        # with evaluation/hit_at_prompt.py (which asks for topk passages the way
        # the app does, instead of slicing a depth-100 list where no cap binds):
        # a flat cap scores 79% of 98 questions at topk=5 against 83% uncapped,
        # losing r03, p02, n02 and n17 - n17's answer is at overall rank 5, and
        # every passage above it is another clause of the same rulebook. The
        # diversity cap exists for the opposite case: "where can I study quietly
        # and borrow books" filled the prompt with one administrative manual. So
        # the cap applies to everything EXCEPT the document family the question
        # is actually about.
        one_doc = scope and (qi["rules"] or qi["course"] or qi["regs"])
        out = []
        for s, i in ranked:
            row = self.meta[i]
            if row.get("duplicate_of"):
                continue
            fam = self.profile[i][0]
            cap_i = topk if (one_doc and fam in ("regulation", "scheme")) else doc_cap
            per_doc = max(self._scope.W["per_doc"].get(fam, 1) if scope else 1, cap_i) * grow
            if src_n[row["source_id"]] >= per_doc:
                continue
            if fam_cap and fam_n[fam] >= max(fam_cap.get(fam, topk), cap_i) * grow:
                continue
            fam_n[fam] += 1
            src_n[row["source_id"]] += 1
            out.append(dict(row, score=round(s, 4)))
            if len(out) >= topk:
                break
        return out


# Three signals, one weight each:
#   keyword   a query term (or an inflection of it) is in the passage
#   synonym   only a WordNet / domain-map synonym of it is
#   semantic  embedding cosine between the question and the passage
# keyword + 0.5 x synonym (semantic 0) is exactly the coverage measure (sir's 1 / 0.5 / 0
# weighting); semantic is a third, non-zero weight added on top of it.
# Override for experiments with VJLM_FUSION="keyword,synonym,semantic".
# Swept on the 98-question set, chosen on dev+hold, reported on test and on the
# 36 never-tuned v2 questions (evaluation/sweep_wordnet.py, sweep_weights.py):
#
#   semantic  1.5 beats 0.5 by 1 point and 0.67x-keyword ratios by 3 (config E)
#   synonym   1.0 beats 0.5 by 2 points, ALL of it on the never-tuned split
#
# The synonym weight now equals the keyword weight, so a synonym match counts as
# much as a literal one. That departs from coverage's 1 / 0.5 / 0 weighting on
# purpose: coverage uses 0.5 to stop a synonym outranking a literal hit, but
# fusion has the semantic arm to break those ties, so the caution costs more
# than it saves. Coverage mode itself is unchanged.
#
# This default used to be "1.0,0.5,0.5" while every evaluation harness
# hardcoded "1.0,0.5,1.5", so the app served a configuration nothing measured.
# The harnesses now only override when VJLM_FUSION is set explicitly.
FUSION = tuple(float(x) for x in os.environ.get("VJLM_FUSION", "1.0,1.0,1.5").split(","))
# BGE is trained to embed a SEARCH QUERY with this instruction in front (passages
# get none); without it short questions sit further from the passages that answer them.
FUSION_QPREFIX = os.environ.get("VJLM_QPREFIX", "")
# 1200, not 300: measured on 98 questions, the deeper semantic pool moved
# hit@20 92% -> 94% and hit@100 95% -> 98% for +60 ms a query, because gold
# chunks sit as deep as cosine rank 1156 (evaluation/dense_recall.py). Past
# 1200 nothing more is reached - 4000 measured identical.
FUSION_DENSE_POOL = int(os.environ.get("VJLM_DENSE_POOL", "1200"))   # semantic first-stage depth
# How many chunks one document may contribute, as a multiple of W["per_doc"].
PER_DOC_SCALE = int(os.environ.get("VJLM_PER_DOC", "1"))
# Both measured and REJECTED on the 98-question set, kept at 0 with the number
# that killed them: a best-sentence rerank of the leaders (hit@5 83% -> 81% at
# weight 1.0, 79% at 2.0 - the gold clause is often a table row with no
# sentence, while the pages above it restate the question in one tidy line), and
# a bonus for section-overview pages (83%, unchanged, and hit@3 77% -> 74%).
IDF_POW = float(os.environ.get("VJLM_IDF_POW", "1.0"))   # rejected: flat 83% at 1.5/2/3
# 5 passages x 1000. 3250 was costing 4 points even after rag/focus.py stopped
# hard-dropping low-scoring sentences: at 650 per passage the answer survived
# into the prompt for 79% of questions, at 1000 for 83% - the full retrieval
# ceiling, so nothing selected is lost any more. 6500 and 9000 measure identical
# (every kept sentence already fits), so this is the smallest budget that leaks
# nothing: 1202 tokens of the 4096 context, leaving room for the answer.
PROMPT_CHARS = int(os.environ.get("VJLM_PROMPT_CHARS", "5000"))
PROMPT_ORDER = os.environ.get("VJLM_PROMPT_ORDER", "score")  # "sent" rejected: identical 23/26
SENT_W = float(os.environ.get("VJLM_SENT", "0"))     # weight of the best-sentence pass
SENT_N = int(os.environ.get("VJLM_SENT_N", "60"))    # how many leaders it re-reads
DOC_CAP_FRAC = float(os.environ.get("VJLM_DOC_CAP_FRAC", "1.0"))  # per-document share of topk
DOC_CAP_MIN = int(os.environ.get("VJLM_DOC_CAP_MIN", "2"))        # never fewer than this
RRF_K = int(os.environ.get("VJLM_RRF", "0"))   # >0 ranks the arms instead of summing
PARTS = None   # set to {} to capture the per-arm breakdown of the last query

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
    # The prompt's character budget is FIXED, so carrying more passages costs no
    # context, no latency and no memory - each one is simply trimmed harder by
    # rag/focus.py, which keeps the sentences that match the question. This is
    # what buys the accuracy: the three signals put the answer in the top 5 for
    # 83% of questions but in the top 10 for 88% and the top 20 for 92%
    # (evaluation/depth_curve.py), and six attempts at reordering the top 20
    # were all rejected by measurement, so the cheaper fix is to show the model
    # more of what retrieval already found.
    budget = max(220, PROMPT_CHARS // max(1, len(chunks)))
    if PROMPT_ORDER == "sent" and len(chunks) > 1:
        # Order the CHOSEN passages by their best question-matching sentence.
        # This cannot change hit@5 - the set is identical - but it changes which
        # passage the model reads first, and this model answers from the early
        # ones: "where can I study quietly and borrow books" had the library
        # page at slot 3 and was answered from the Administrative Manual at
        # slots 1-2. Reranking the POOL by best sentence was measured and
        # rejected (hit@5 83% -> 81%); ordering what retrieval already chose is
        # a different question, and only the end-to-end test can answer it.
        from rag.focus import scored_units
        def _best(c):
            su = scored_units(query, c["text"], IDF)
            return max((s for s, _, _ in su), default=0.0)
        chunks = sorted(chunks, key=_best, reverse=True)
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
        # The heading path goes in the HEADER line, never in the body. Retrieval
        # indexes "document > section > sub-section" together with the text
        # (rag/index.py:index_text) and the model needs it too: several branches'
        # syllabus tables read identically, and a clause means little without the
        # section it sits under. It stays out of the body because this model
        # answers by copying the passage's first sentence, and a breadcrumb is
        # not an answer.
        where = re.sub(r"\s+", " ", (c.get("context") or "")).strip()
        if where:
            meta += " " + where[:120]
        parts.append("[{}] {} (page {}){}\n{}".format(
            i + 1, c["url"], c["page"], meta,
            redact(_focus(query, c["text"], IDF, max_chars=budget))))
    ctx = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM + "\n" + ROLE_NOTE.get(role, "")},
        {"role": "user", "content": "CONTEXT:\n{}\n\nQUESTION: {}".format(ctx, query)},
    ]
