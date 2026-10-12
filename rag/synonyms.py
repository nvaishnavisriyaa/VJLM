"""WordNet query expansion for the lexical arm of hybrid retrieval.

Why this exists
---------------
BM25 matches surface tokens. A student asks "what is the attendance
requirement"; the regulation says "a candidate shall put in a minimum of 75% of
attendance". The shared token is "attendance"; "requirement" contributes
nothing, and a question phrased "rules for attendance shortage" shares almost
nothing with "condonation of shortage of attendance". Expanding each query term
with its WordNet synonyms gives the lexical arm a chance at those.

Why it is NOT free
------------------
Query expansion is a classic recall/precision trade. WordNet is a general
English lexicon with no idea it is being pointed at academic regulations, so
"term" expands to "condition", "word" and "full term", and most senses of a
word are irrelevant in this domain. Naive expansion reliably makes retrieval
worse. Three guards, in order of how much they matter:

1. **Vocabulary gating.** A synonym is only kept if it actually occurs in the
   indexed corpus (`vocab`). Most WordNet noise never appears in VNRVJIET
   documents, so this alone removes the bulk of it and costs nothing.
2. **Weight discount.** Expanded terms enter BM25 at `weight` (default 0.35)
   against 1.0 for terms the user actually typed. An expansion can promote a
   document that would have ranked just below the cut; it cannot outvote a
   literal match.
3. **Budget.** At most `max_per_term` synonyms per query term, ordered by
   WordNet sense frequency, so the common sense wins over the obscure one.

Derivational forms ("attend" <-> "attendance", "admit" <-> "admission") are
included because academic prose switches between verb and noun forms
constantly, and they are far safer than loose synonyms - they are the same
lemma, not a different concept.

This is off by default. Turn it on only if `evaluation/eval_retrieval.py` says
it helps on the VJLM benchmark - see `RAG-SYN-001` in DECISIONS.md.
"""
import functools
import os
import re

_WN = None
_UNAVAILABLE = None

# Domain terms whose WordNet senses are actively misleading here. "term" ->
# "condition"/"word", "credit" -> "recognition"/"deferred payment",
# "class" -> "social class", "board" -> "plank". Cheaper to block a handful of
# known-bad expansions than to accept the noise.
BLOCK = {
    "term", "credit", "class", "board", "grade", "year", "paper", "course",
    "program", "programme", "project", "practice", "head", "chair", "college",
    "institute", "department", "student", "faculty", "degree", "branch",
    # Function-ish words the probe caught expanding into nonsense:
    # do -> bash/ut, about -> approximately, low -> depression, batch -> heap.
    "do", "does", "get", "got", "make", "take", "put", "go", "come", "give",
    "about", "many", "much", "need", "want", "low", "high", "batch", "lot",
    "thing", "way", "time", "part", "number", "one", "two", "first", "last",
}

# Academic pairs WordNet does not link but this domain treats as equivalent.
# Kept deliberately small and auditable - this is a hand-maintained list, not a
# thesaurus, and every entry should be defensible to an examiner.
DOMAIN = {
    "attendance": ["attend", "presence", "shortage"],
    "detention": ["detained", "shortage", "condonation"],
    "condonation": ["condone", "shortage", "attendance"],
    "regulation": ["rule", "ordinance", "regulations"],
    "syllabus": ["curriculum", "course", "scheme"],
    "fee": ["fees", "tuition", "payment"],
    "fees": ["fee", "tuition", "payment"],
    "placement": ["recruitment", "hiring", "placements"],
    "hostel": ["accommodation", "residence"],
    "scholarship": ["stipend", "financial"],
    "exam": ["examination", "exams", "test"],
    "exams": ["examination", "exam", "test"],
    "examination": ["exam", "test", "assessment"],
    "backlog": ["arrear", "arrears", "supplementary"],
    "cgpa": ["gpa", "grade", "sgpa"],
    "hod": ["head", "chairperson"],
    "timetable": ["schedule", "calendar"],
    "eligibility": ["eligible", "criteria", "requirement"],
    "requirement": ["required", "criteria", "minimum"],
    # Student vocabulary -> the regulations' vocabulary (RAG-SCOPE-001). General
    # phrasing students use, not tuned to any one test question.
    "clear": ["pass", "passed", "passing"],
    "pass": ["clear", "passed", "passing"],
    "subject": ["course", "courses"],
    "subjects": ["course", "courses"],
    "repeat": ["supplementary", "reappear", "re-appear"],
    "miss": ["absent", "absence"],
    "missed": ["absent", "absence"],
    # Students say "sit"/"write" an exam; the regulations say "appear for" it.
    # Measured consequence: in "can I still sit for the final exams", `sit`
    # (idf 7.49) and `exams` (6.69) are RARER in this corpus than `attendance`
    # (5.25), because the rulebook never uses the student's words - so the
    # casual phrasing, not the topic, looked like the discriminating term.
    "sit": ["appear", "write", "attend"],
    "write": ["appear", "sit"],
    "sitting": ["appearing", "appear"],
    "challenge": ["revaluation", "recounting"],
    "recheck": ["revaluation", "recounting"],
    "complain": ["grievance", "complaint", "committee"],
    "complaint": ["grievance", "committee"],
    "founded": ["established", "establishment"],
    "established": ["founded", "establishment"],
    "salary": ["package", "ctc", "lpa"],
    "package": ["ctc", "lpa", "salary"],
    "marks": ["grade", "grades", "score"],
    "final": ["semester", "end"],
    "affiliated": ["affiliation"],
}

_WORD = re.compile(r"^[a-z][a-z0-9]*$")


def _wordnet():
    """Load WordNet once. Returns None if unavailable - expansion then
    degrades to the domain map rather than raising, because retrieval working
    slightly worse is better than retrieval not working."""
    global _WN, _UNAVAILABLE
    if _WN is not None or _UNAVAILABLE:
        return _WN
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")          # force the corpus load here, not mid-query
        _WN = wn
    except Exception as e:          # noqa: BLE001
        _UNAVAILABLE = str(e)[:120]
        print("  [synonyms] WordNet unavailable (%s); using domain map only"
              % _UNAVAILABLE)
    return _WN


def available():
    return _wordnet() is not None


# Measured 2026-09-26 (evaluation/paraphrase_probe.py): unrestricted WordNet
# expansion was overwhelmingly noise on this corpus. Verbs were the worst -
# "attend" -> serve, "sit" -> posture, "miss" -> girl, "do" -> bash - because
# WordNet ranks senses over general English, not over academic regulations,
# and a query verb almost never carries the retrievable content here. Two
# restrictions follow from that evidence:
#
#   * NOUNS ONLY. The content of an academic query lives in its nouns
#     (attendance, regulation, condonation, placement). Verbs and adjectives
#     are where the wrong-sense expansions came from.
#   * FIRST SENSE ONLY (MAX_SYNSETS). WordNet orders synsets by frequency, so
#     synset 0 is the everyday meaning. Later synsets are where "miss" becomes
#     "girl". Derivational forms are still taken from all kept synsets because
#     they are the same lemma, not a different concept.
MAX_SYNSETS = 1
NOUNS_ONLY = True


@functools.lru_cache(maxsize=8192)
def _raw_synonyms(term):
    """WordNet lemmas + derivational forms for one term, sense-ordered.

    Multi-word lemmas ("rule_of_law") are dropped: BM25 indexes single tokens,
    so they can never match anyway.
    """
    wn = _wordnet()
    if wn is None:
        return ()
    out, seen = [], {term}
    try:
        pos = wn.NOUN if NOUNS_ONLY else None
        synsets = wn.synsets(term, pos=pos) if pos else wn.synsets(term)
    except Exception:               # noqa: BLE001
        return ()
    for syn in synsets[:MAX_SYNSETS]:   # sense-frequency ordered; 0 is common
        for lem in syn.lemmas():
            name = lem.name().lower()
            if "_" in name or not _WORD.match(name) or name in seen:
                continue
            seen.add(name)
            out.append(name)
            for rel in lem.derivationally_related_forms():
                d = rel.name().lower()
                if "_" not in d and _WORD.match(d) and d not in seen:
                    seen.add(d)
                    out.append(d)
    return tuple(out)


class Expander:
    """Expands a tokenised query into weighted terms.

    vocab: the indexed vocabulary (BM25.idf keys). Synonyms outside it are
    discarded - they cannot match any document, so keeping them only costs
    time and invites drift.
    """

    def __init__(self, vocab=None, weight=0.35, max_per_term=3,
                 use_wordnet=True, use_domain=True):
        self.vocab = vocab
        self.weight = weight
        self.max_per_term = max_per_term
        self.use_wordnet = use_wordnet
        self.use_domain = use_domain
        self.last = {}              # term -> [kept synonyms], for explainability

    def _ok(self, w):
        return self.vocab is None or w in self.vocab

    def expand_term(self, term):
        if term in BLOCK:
            return []
        cands = []
        if self.use_domain:
            cands.extend(DOMAIN.get(term, []))
        if self.use_wordnet:
            cands.extend(_raw_synonyms(term))
        kept, seen = [], {term}
        for c in cands:
            if c in seen or c in BLOCK or not self._ok(c):
                continue
            seen.add(c)
            kept.append(c)
            if len(kept) >= self.max_per_term:
                break
        return kept

    def expand(self, tokens):
        """-> [(token, weight)], originals at 1.0, synonyms at self.weight.

        A synonym reached from two different query terms is kept once, at the
        higher weight, so it cannot be double-counted into outranking a
        literal match.
        """
        weights = {}
        self.last = {}
        for t in tokens:
            weights[t] = 1.0
        for t in tokens:
            syns = self.expand_term(t)
            if syns:
                self.last[t] = syns
            for s in syns:
                if weights.get(s, 0.0) < self.weight:
                    weights[s] = self.weight
        return sorted(weights.items(), key=lambda kv: -kv[1])

    def explain(self):
        if not self.last:
            return "no expansions"
        return "; ".join("%s -> %s" % (k, ", ".join(v)) for k, v in self.last.items())


def ensure_corpus(quiet=True):
    """Download the WordNet corpus if missing. Build-time only - a one-off
    data fetch, never part of serving a query."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("test")
            return True
        except LookupError:
            nltk.download("wordnet", quiet=quiet)
            nltk.download("omw-1.4", quiet=quiet)
            wn.synsets("test")
            return True
    except Exception as e:          # noqa: BLE001
        print("  [synonyms] could not prepare WordNet: %s" % str(e)[:120])
        return False


if __name__ == "__main__":
    ensure_corpus(quiet=False)
    ex = Expander(vocab=None, max_per_term=4)
    for q in ["what is the attendance requirement",
              "rules for detention and condonation",
              "exam timetable for third year",
              "placement statistics and recruitment"]:
        toks = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) > 2]
        print("\nQ:", q)
        for t, w in ex.expand(toks):
            print("   %-16s %.2f" % (t, w))
