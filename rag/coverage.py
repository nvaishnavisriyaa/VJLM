"""Synonym-aware coverage similarity.

The measure
-----------
For each query term, a document scores:

    1.0   the term itself occurs in the document
    0.5   a synonym of it occurs (WordNet or the curated domain map)
    0.0   neither

Those per-term hits are combined into one similarity in [0, 1].

Why IDF-weight the combination
------------------------------
A plain average treats every query term as equally informative, so in
"what is the attendance requirement for students" the words `students` and
`attendance` count the same. `students` occurs in most of the corpus and
separates nothing; `condonation` occurs in a handful of documents and
separates almost perfectly. Weighting each term by its IDF makes the measure
reward covering the *discriminating* terms:

    coverage = sum_t ( hit(t) * idf(t) ) / sum_t idf(t)

The denominator is the score a document would get by containing every query
term literally, so the result stays a 0-1 fraction and is comparable across
queries of different lengths.

Why this is worth having alongside BM25
---------------------------------------
They disagree, and the disagreement is the point.

* **BM25** is a term-frequency measure with saturation and length
  normalisation. One high-IDF term repeated often can dominate the score, so
  a document about *only* attendance ranks top for a query about attendance
  rules during examinations.
* **Coverage** is a breadth measure. It cannot be won by repetition - a term
  either contributes its IDF or it does not. A document touching attendance,
  examinations *and* condonation beats one that mentions attendance fifty
  times.

For multi-concept questions - which is what students actually ask - breadth
is usually the better signal, and fusing the two beats either alone. That is
an empirical claim, so it is measured in `evaluation/eval_retrieval.py`
(modes `coverage` and `hybrid`) rather than asserted here.

Cost
----
Scored only over the candidate pool a cheaper retriever already produced
(~60 documents), never the full 13k corpus. Presence tests are O(1) against
the BM25 term-frequency counters that already exist, so this adds no index
and no extra memory.
"""


class CoverageScorer:
    """Scores candidates by synonym-aware, IDF-weighted query coverage.

    bm25: a built BM25 (needs .tf, one Counter per document, and .idf).
    expander: rag.synonyms.Expander, or None for exact-match-only coverage.
    """

    #: score for a literal term match
    W_EXACT = 1.0
    #: score for a synonym match - half credit, per the rule this implements
    W_SYNONYM = 0.5

    def __init__(self, bm25, expander=None, w_exact=None, w_syn=None):
        self.tf = bm25.tf
        self.idf = bm25.idf
        self.expander = expander
        self.w_exact = self.W_EXACT if w_exact is None else w_exact
        self.w_syn = self.W_SYNONYM if w_syn is None else w_syn
        # An unseen query term has no IDF. Use the maximum observed IDF: a term
        # absent from the whole corpus is maximally rare, so it belongs at the
        # top of the scale, not the bottom. It can never be matched, so it only
        # ever enters the denominator - correctly penalising documents for not
        # containing something distinctive the user asked about.
        self._max_idf = max(self.idf.values()) if self.idf else 1.0

    def _idf(self, term):
        return self.idf.get(term, self._max_idf)

    def build_terms(self, tokens):
        """-> [(term, idf, [synonyms])], one entry per distinct query token."""
        out, seen = [], set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            syns = self.expander.expand_term(t) if self.expander else []
            out.append((t, self._idf(t), syns))
        return out

    def score_doc(self, terms, doc_index):
        """Coverage of one document, in [0, 1]."""
        tf = self.tf[doc_index]
        num = 0.0
        den = 0.0
        for term, idf, syns in terms:
            den += idf
            if term in tf:
                num += self.w_exact * idf
            elif syns:
                for s in syns:
                    if s in tf:
                        num += self.w_syn * idf
                        break
        return (num / den) if den else 0.0

    def score_many(self, tokens, candidates):
        """-> {doc_index: coverage} over the candidate pool only."""
        terms = self.build_terms(tokens)
        if not terms:
            return {}
        return {i: self.score_doc(terms, i) for i in candidates}

    def explain(self, tokens, doc_index):
        """Per-term breakdown, for showing why a document was retrieved."""
        tf = self.tf[doc_index]
        rows = []
        for term, idf, syns in self.build_terms(tokens):
            hit, via = 0.0, None
            if term in tf:
                hit, via = self.w_exact, term
            elif syns:
                for s in syns:
                    if s in tf:
                        hit, via = self.w_syn, s
                        break
            rows.append({"term": term, "idf": round(idf, 3),
                         "weight": hit, "matched_via": via})
        den = sum(r["idf"] for r in rows) or 1.0
        total = sum(r["weight"] * r["idf"] for r in rows) / den
        return {"coverage": round(total, 4), "terms": rows}
