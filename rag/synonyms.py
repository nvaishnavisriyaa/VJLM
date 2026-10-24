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
    # A second pass over the vocabulary gap, from evaluation/miss_report.py: the
    # gold clause used none of the question's words, so neither BM25 nor
    # coverage could reach it. "orientation programme" is called an INDUCTION
    # programme; "the lowest mark that still passes" is "a MINIMUM of 35%";
    # "exams for the subjects I failed" are SUPPLEMENTARY examinations; "how
    # many seats" is the SANCTIONED INTAKE. These are the institution's words
    # for the student's words, not bridges built for particular questions.
    "orientation": ["induction", "inducting"],
    "induction": ["orientation"],
    "lowest": ["minimum", "least"],
    "least": ["minimum", "lowest"],
    "minimum": ["least", "lowest"],
    "failed": ["fail", "supplementary", "backlog"],
    "fail": ["failed", "supplementary", "backlog"],
    "supplementary": ["backlog", "arrear", "reappear"],
    "seats": ["intake", "sanctioned", "seat"],
    "seat": ["intake", "sanctioned"],
    "intake": ["seats", "sanctioned"],
    "revaluation": ["recounting", "recheck"],
    "club": ["clubs", "society", "chapter"],
    "clubs": ["club", "societies", "society"],
    "society": ["societies", "club", "clubs"],
    "societies": ["society", "clubs", "club"],
    # The academic calendars are the clearest vocabulary gap in the corpus: a
    # student asks when classes START for SECOND year, and the calendar says
    # "Commencement of Class Work" for "B.Tech. II Year". Both of the calendar
    # questions that fail (the end-to-end Q20 and d02) fail on exactly this,
    # with the right page extracted and the dates present in it. `commencement`
    # is rare in this corpus (IDF 5.91), so it discriminates strongly once the
    # query can reach it, and the Roman-numeral year is how every VNRVJIET
    # document names a year.
    "start": ["commencement", "commences", "begin"],
    "starts": ["commencement", "commences", "begin"],
    "begin": ["commencement", "commences", "start"],
    "begins": ["commencement", "commences", "start"],
    "commence": ["start", "begin"],
    "classes": ["class", "classwork", "instruction"],
    "second": ["ii"],
    "third": ["iii"],
    "fourth": ["iv"],
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
# NOUNS_ONLY threw out every verb to stop a handful of bad senses, and measured
# on the 98-question set the whole WordNet arm then contributed exactly NOTHING:
# Hit@5 was 83% with it on and 83% with it off (evaluation/ablation.py), because
# the questions that need a synonym need a VERB one - "when do classes START" ->
# "COMMENCEMENT of classes", "can I get my marks RECHECKED" -> "REVALUATION",
# "exams for FAILED subjects" -> "SUPPLEMENTARY examinations". `start` returned
# [] while WordNet's start.v.01 holds exactly the lemmas `begin` and `commence`.
#
# But the original evidence against verbs was real, so the fix is a sense filter
# rather than lifting the restriction. Look at WHICH senses were wrong - "miss"
# -> girl, "attend" -> serve, "club" -> nine, "seat" -> place. Every one is a
# COMMON word; every synonym that helps - commencement, revaluation,
# condonation, supplementary - is RARE. So gate on IDF: keep a WordNet synonym
# only when it is at least as discriminating in THIS corpus as the term it came
# from. A wrong sense is generic by nature and fails the gate; a rare technical
# term passes. That makes the arm corpus-grounded instead of general-English.
MAX_SYNSETS = int(os.environ.get("VJLM_WN_SYNSETS", "1"))
# Which parts of speech to expand: n=noun, v=verb, a=adjective, r=adverb.
# "nv" and "n" measured byte-identical on all 98 questions at every gate and cap
# (evaluation/sweep_wordnet.py), so verbs cost nothing; they are allowed because
# the gate now makes them safe, not because they were shown to pay.
WN_POS = os.environ.get("VJLM_WN_POS", "nv")
# Keep synonym s for term t only if idf(s) >= ratio * idf(t). 0 disables.
# 1.0 measured +1 point (83% -> 84/98), entirely on the never-tuned split;
# 1.3 measured identical, so 1.0 is the loosest gate that captures the gain.
WN_IDF_RATIO = float(os.environ.get("VJLM_WN_IDF", "1.0"))


def _pos_list(wn):
    codes = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}
    return [codes[c] for c in WN_POS if c in codes] or [wn.NOUN]


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
    synsets = []
    for pos in _pos_list(wn):
        try:
            # Per POS, not pooled: pooling lets a noun's senses crowd out the
            # verb reading entirely, which is how `start` came back empty.
            synsets.extend(wn.synsets(term, pos=pos)[:MAX_SYNSETS])
        except Exception:           # noqa: BLE001
            continue
    if not synsets:
        return ()
    for syn in synsets:                 # sense-frequency ordered; 0 is common
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

    def _idf_ok(self, term, syn):
        """Is `syn` at least as discriminating as `term` in this corpus?

        The vocabulary is BM25's idf MAP, so the values are here already. A
        synonym the gate cannot judge (either side missing) is allowed through -
        the vocab check has already established it occurs in the corpus.
        """
        if not WN_IDF_RATIO:
            return True
        try:
            ti = self.vocab.get(term)
            si = self.vocab.get(syn)
        except AttributeError:      # a plain set, no idf values to gate on
            return True
        if ti is None or si is None:
            return True
        return si >= WN_IDF_RATIO * ti

    def expand_term(self, term):
        if term in BLOCK:
            return []
        cands = []                  # (word, came_from_wordnet)
        if self.use_domain:
            # The map is keyed by the lemma, but students type the inflection:
            # "rechecked" found nothing while "recheck" -> revaluation was
            # sitting right there. Same lemma, so this is a lookup fix, not an
            # extra hop through meaning.
            cands.extend((w, False) for w in DOMAIN.get(term, []))
            if term not in DOMAIN:
                from rag.scope import variants
                for v in variants(term):
                    if v in DOMAIN:
                        cands.extend((w, False) for w in DOMAIN[v])
                        break
        if self.use_wordnet:
            cands.extend((w, True) for w in _raw_synonyms(term))
        kept, seen = [], {term}
        for c, from_wn in cands:
            if c in seen or c in BLOCK or not self._ok(c):
                continue
            # The IDF gate applies to WordNet only. The domain map is hand-built
            # from this corpus and is trusted; WordNet is general English and is
            # where the wrong senses come from.
            if from_wn and not self._idf_ok(term, c):
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
