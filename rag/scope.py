"""Query scoping and source priors (RAG-SCOPE-001).

Two failures dominated the first student test (42% answer accuracy, 31% hit@5):

1. WRONG REGULATION. "B.Tech R25" questions retrieved R18/R19/R22 and M.Tech
   rulebooks, and the model answered confidently from them ("O grade = 4",
   CIE = 30+10). Every rulebook says nearly the same words, so word matching
   cannot tell them apart - the question's R-code and programme can.
2. WRONG KIND OF SOURCE. 62% of chunks are syllabus text and ~5% faculty CVs,
   which share vocabulary with everything. A student asking about attendance,
   NAAC or the library almost never wants a CV or a 2012 lab manual.

This module reads the query (regulation code, programme, intent) and each
document's profile (family, regulation, programme) and returns a multiplier.
It is deliberately small and rule-based: no model, inspectable, and every
constant below is a stated assumption recorded in DECISIONS.md, tuned on the
dev split only.
"""
import math
import os
import re

# ---------------------------------------------------------------- documents
CURRENT_REG = "R25"
# Regulations whose rulebooks/schemes are still in force for enrolled students.
ACTIVE_REGS = {"R25", "R22"}

_REG_FILE = [  # (url regex, regulation, programme)
    (r"vnr-btech-r25-regulations", "R25", "btech"),
    (r"vnr-r22-btech-academic-regulations", "R22", "btech"),
    (r"vnrmtr25", "R25", "mtech"),
    (r"vnrmtr22", "R22", "mtech"),
    (r"vnr-mca-r25", "R25", "mca"),
    (r"vnrr19", "R19", "btech"),
    (r"vnrr18", "R18", "btech"),
    (r"vnra19", "A19", "btech"),
    (r"vnrmta18", "A18", "mtech"),
    (r"vnrmt18", "R18", "mtech"),
]
_REG_FILE = [(re.compile(a), b, c) for a, b, c in _REG_FILE]
_REG_IN_URL = re.compile(r"(?<![a-z0-9])([ra]\d{2})(?![0-9])", re.I)

FAMILIES = ("regulation", "scheme", "site", "report", "faculty", "notice", "other")


# Branch (department) codes as they appear in a URL or file name:
# II-B.Tech-CSBS-R25.pdf, /departments-ae, /cse-aiml/... The longest alias has
# to win, so these are ordered longest-first and joined into one alternation.
_BRANCH_CODES = ("cse-aiml", "cse-ds", "csbs", "aids", "biotech", "civil", "mech",
                 "vlsi", "mca", "mba", "ece", "eee", "cys", "iot", "rai", "cse",
                 "ei", "ae")
_BRANCH_URL = re.compile(r"(?<![a-z])(" + "|".join(_BRANCH_CODES) + r")(?![a-z])")

# What a student calls the branch -> the code the documents use. Spelled-out
# names come first so "computer science and business systems" is not read as
# "computer science". `it` is deliberately absent: as a query token it is the
# pronoun far more often than the branch.
_BRANCH_Q = [
    ("computer science and business systems", "csbs"),
    ("artificial intelligence and data science", "aids"),
    ("artificial intelligence and machine learning", "cse-aiml"),
    ("electronics and communication", "ece"),
    ("electrical and electronics", "eee"),
    ("electronics and instrumentation", "ei"),
    ("internet of things", "iot"),
    ("cyber security", "cys"),
    ("data science", "cse-ds"),
    ("automobile", "ae"),
    ("mechanical", "mech"),
    ("biotechnology", "biotech"),
    ("instrumentation", "ei"),
    ("robotics", "rai"),
    ("civil", "civil"),
    ("csbs", "csbs"), ("cse-aiml", "cse-aiml"), ("aiml", "cse-aiml"),
    ("cse-ds", "cse-ds"), ("aids", "aids"), ("vlsi", "vlsi"), ("cys", "cys"),
    ("iot", "iot"), ("ece", "ece"), ("eee", "eee"), ("mech", "mech"),
    ("cse", "cse"), ("mca", "mca"), ("mba", "mba"),
]


def query_branch(ql):
    """The branch a question names, or None. Longest alias wins."""
    for alias, code in _BRANCH_Q:
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", ql):
            return code
    return None


def doc_profile(row):
    """-> (family, regulation|None, programme|None, depth, branch|None).
    depth = number of path segments for site pages (/about = 1, /campus-life/clubs/x = 3)."""
    fam, reg, prog = _doc_profile(row)
    depth = 0
    if fam == "site":
        depth = len([x for x in re.sub(r"^https?://[^/]+", "", row.get("url") or "").split("/") if x])
    # Branch from the enriched metadata when it is there (38k chunks carry it),
    # otherwise from the file name, which is where a syllabus PDF keeps its only
    # mention of the branch - its course tables never repeat it.
    branch = (row.get("department") or "").lower() or None
    if not branch:
        m = _BRANCH_URL.search((row.get("url") or "").lower())
        branch = m.group(1) if m else None
    return fam, reg, prog, depth, branch


def _doc_profile(row):
    url = (row.get("url") or "").lower().replace("%20", " ")
    cat = row.get("category") or ""
    for rx, reg, prog in _REG_FILE:
        if rx.search(url):
            return "regulation", reg, prog
    reg = row.get("regulation")
    if not reg:
        m = _REG_IN_URL.search(url)
        reg = m.group(1).upper() if m else None
    prog = None
    if re.search(r"m\.?tech|mtech", url):
        prog = "mtech"
    elif "mca" in url:
        prog = "mca"
    elif re.search(r"b\.?tech|-r\d\d|/r\d\d/|vnrr|vnra", url):
        prog = "btech"
    if cat == "faculty" or re.search(r"/faculty/|knowledge asset|-cv\b|-ka-\d{4}", url):
        return "faculty", reg, prog
    if re.search(r"/iqac/|aqar|nirf|ssr\b|naac|/aicte/|annual-report|iiqa", url):
        return "report", reg, prog
    if row.get("document_type") == "html" and "://www.vnrvjiet.ac.in" in url:
        return "site", reg, prog
    if cat == "syllabus" or re.search(r"-r2\d\.pdf|/r\d\d/|mtechsyl|scheme", url):
        return "scheme", reg, prog
    if cat == "academic_regulations":
        return "regulation", reg, prog
    if "docs.vnrvjiet.ac.in" in url or "vnrvjiet.ac.in/docs" in url:
        return "notice", reg, prog
    return "other", reg, prog


# -------------------------------------------------------------------- query
_RULES = re.compile(
    r"attendance|condon|detain|credit|grade|cgpa|sgpa|\bcie\b|\bsee\b|marks?\b|"
    r"pass(ing)?\b|fail|supplementary|revaluation|promot|backlog|regulation|"
    r"semester end|mid exam|sessional|eligib|minimum|maximum|award of|degree",
    re.I)
_PERSON = re.compile(r"\bwho is\b|professor|\bdr\.?\b|\bhod\b|head of the department|"
                     r"faculty (member|profile)|publications|\bresume\b|\bcv\b", re.I)
_REPORT = re.compile(r"\baqar\b|\bnirf\b|\bssr\b|annual (quality|report)|iqac|\bnba\b", re.I)
_COURSE = re.compile(r"syllabus|\b\d{2}[a-z]{2}\d[a-z]{2}\d{3}\b|course outcomes?|"
                     r"subjects?\b|curriculum|scheme|credits? (of|for|is)|lab(oratory)?\b", re.I)


# "What does SGPA stand for?" is answered by the abbreviation table, a chunk
# that mentions SGPA once, never by the many chunks that use it repeatedly.
_DEFINE = re.compile(r"\b(?:stands?\s+for|full\s+form|abbreviation|acronym|expansion of|expand(?:ed)?)\b", re.I)
_GLOSSARY = re.compile(r"Abbreviation\s+Full Form|Acronym\s+Expansion|^\s*\d*\s*Definitions\b", re.I)
# A glossary row: an acronym followed by its words ("SGPA Semester Grade Point
# Average"). A table can run over several chunks and only the first carries the
# "Abbreviation Full Form" header, so the rows themselves are the evidence.
_GLOSS_ROW = re.compile(r"^\s*[A-Z][A-Za-z]{0,3}[A-Z][A-Za-z]{0,4}\s+[A-Z][a-z]+(?:\s+[A-Za-z&/()\-]+){1,8}\s*$", re.M)


def parse_query(q):
    ql = q.lower()
    regs = {m.group(1).upper() for m in re.finditer(r"(?<![a-z0-9])([ra]\s?-?\d{2})(?![0-9])", ql)}
    regs = {r.replace(" ", "").replace("-", "") for r in regs}
    prog = None
    if re.search(r"m\.?\s?tech|mtech", ql):
        prog = "mtech"
    elif re.search(r"\bmca\b", ql):
        prog = "mca"
    elif re.search(r"b\.?\s?tech|btech|undergrad|\bug\b", ql):
        prog = "btech"
    return {
        "regs": regs, "prog": prog,
        "rules": bool(_RULES.search(q)),
        "person": bool(_PERSON.search(q)),
        "report": bool(_REPORT.search(q)),
        "course": bool(_COURSE.search(q)),
        "define": bool(_DEFINE.search(q)),
        "branch": query_branch(ql),
        "listy": bool(_LISTY.search(q)),
    }


# ----------------------------------------------------------------- weights
# Every number is an assumption, listed in DECISIONS.md RAG-SCOPE-001.
W = dict(
    reg_match=1.6, reg_mismatch=0.35,           # question names a regulation
    reg_current=1.25, reg_active=1.0, reg_old=0.6,   # question names none
    prog_mismatch=0.4,                           # question names a programme
    prog_default_other=0.6,                      # question names none: students are mostly B.Tech
    faculty=0.4, report=0.6, scheme=0.7,         # priors when the intent is not theirs
    site=1.25,                                   # curated, current pages
    head=0.30,                                   # weight of section-heading coverage
)


def factor(profile, qi):
    fam, reg, prog, depth, branch = profile
    f = 1.0
    # A question that names a branch is answered by that branch's document. The
    # syllabus PDFs of CSBS, IOT, CYS and AIDS are near-identical in wording, so
    # the only thing separating them is the branch - and "how many credits is
    # Design and Analysis of Algorithms in CSBS" used to put all four in the
    # prompt and be answered from the wrong one. Same shape as the regulation
    # mismatch above: a document of ANOTHER branch is penalised, a document of
    # no particular branch is left alone.
    if qi.get("branch") and branch:
        f *= W["branch_match"] if branch == qi["branch"] else W["branch_mismatch"]
    if fam == "regulation" or (fam == "scheme" and (qi["rules"] or qi["regs"])):
        if qi["regs"]:
            if reg in qi["regs"]:
                f *= W["reg_match"]
            elif reg:
                f *= W["reg_mismatch"]
        elif reg:
            f *= (W["reg_current"] if reg == CURRENT_REG
                  else W["reg_active"] if reg in ACTIVE_REGS else W["reg_old"])
        if prog:
            if qi["prog"] and prog != qi["prog"]:
                f *= W["prog_mismatch"]
            elif not qi["prog"] and prog in ("mtech", "mca"):
                f *= W["prog_default_other"]
    if fam == "faculty" and not qi["person"]:
        f *= W["faculty"]
    elif fam == "report" and not qi["report"]:
        f *= W["report"]
    elif fam == "scheme" and not (qi["course"] or qi["rules"] or qi["regs"]):
        f *= W["scheme"]
    elif fam == "site":
        f *= W["site"] * (W["site_top"] if depth <= 1 else 1.0)
    return f


def quota(qi, topk):
    """Max chunks per family in the final top-k. Keeps one noisy family from
    filling every slot; relaxed when the question is genuinely about it."""
    q = {"faculty": 1, "report": 1, "scheme": 2, "notice": max(2, topk - 2)}
    if qi["person"]:
        q["faculty"] = topk
    if qi["report"]:
        q["report"] = topk
    if qi["course"]:
        q["scheme"] = topk
    return q


# ------------------------------------------------------------ chunk quality
# Words that identify WHICH document, not what it says. Scope already uses them
# to pick the rulebook; counting them again at chunk level makes the cover page
# and table of contents (which repeat them) beat the section that answers.
ID_TERMS = frozenset("""tech btech mtech mca regulation regulations r25 r22 r19 r18 r15 a19 a18
    vnrvjiet vnr vjiet college institute student students""".split())

_TOC_HEAD = re.compile(r"table of contents|part no\.? section title", re.I)
_TOC_LINE = re.compile(
    r"\b\d{1,2}\.0\s+[A-Z][A-Za-z ,&/\-]{3,70}?\s\d{1,3}(?:-\d{1,3})?(?=\s+\d{1,2}\.0\s|\s+PART\b|\s*$)")


def is_toc(text):
    """Table of contents / index page: lists every heading, answers nothing."""
    return bool(_TOC_HEAD.search(text)) or len(_TOC_LINE.findall(text)) >= 3


# Words that carry the SHAPE of a question, not its subject. The index drops
# only classic stop words, so "does", "many", "have" survive tokenising - and
# because IDF measures rarity in a rulebook that says "shall" where students
# say "does", they come out RARE: "does" 5.33, "many" 5.02, "have" 3.47, against
# "naac" 4.25 and "attendance" 5.25. "What NAAC grade does VNRVJIET have?"
# then ranked a grading-rules chunk (grade + does + have = 0.75 coverage) above
# the accreditation page that actually says NAAC. Dropped from the QUERY only;
# the index is untouched, so nothing needs re-indexing.
QUESTION_SCAFFOLD = frozenset("does did many much have has there stand stands mean means".split())


def content_terms(tokens):
    """Query tokens minus document-identity words and question scaffolding
    (kept if nothing else is left)."""
    kept = [t for t in tokens if t not in ID_TERMS and t not in QUESTION_SCAFFOLD]
    return kept if kept else [t for t in tokens if t not in ID_TERMS] or tokens


# ------------------------------------------- short, case-bearing identifiers
# rag.index.tok lowercases and drops single characters, so the question
# "what grade point is given for the O grade in B.Tech R25?" is searched as
# [grade, point, given] - the O is gone, and every grade table in every
# regulation covers those terms equally well. "SEE" (Semester End Examination)
# collapses into the common verb "see" the same way. Both are recovered from
# the RAW query and matched against RAW chunk text with case intact, where
# they are discriminating again. Rebuilding the BM25 index with a tokenizer
# that keeps them would cost a full re-index of 104k chunks; this is a
# rescoring pass over the candidate pool coverage already built.
_STOP_CAPS = frozenset("I A AN THE AND OR OF IN ON AT TO IS IT BE AS BY FOR DO HOW WHAT".split())
_ACRO = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
_GRADE_LETTER = re.compile(r"\b([A-Z][+-]?)[\s-]+grade\b|\bgrade[\s-]+([A-Z][+-]?)\b")


def identifiers(raw_query):
    """-> [(case-sensitive pattern, label)] for the query's short identifiers.

    Regulation codes (R25, A19) are deliberately excluded: `factor` already
    scopes the document family by regulation, and matching them at chunk level
    promotes cover pages that merely repeat the code.
    """
    out, seen = [], set()

    def add(text):
        if text not in seen:
            seen.add(text)
            out.append((re.compile(r"\b%s\b" % re.escape(text)), text))

    for m in _GRADE_LETTER.finditer(raw_query):
        g = m.group(1) or m.group(2)
        if g and len(g.rstrip("+-")) == 1:
            add(g)
    for m in _ACRO.finditer(raw_query):
        a = m.group(0)
        if (a in _STOP_CAPS or a.lower() in ID_TERMS
                or a.rstrip("0123456789") in ("R", "A")):
            continue
        add(a)
    return out


def id_weights(ids, idf):
    """[(pattern, idf)] - an identifier counts for what it discriminates.

    A flat bonus per identifier was measured and rejected: it let the roman
    numeral in "II year B.Tech" (idf 1.65, it appears in Unit II, Table II,
    II semester throughout the corpus) outweigh real coverage, and the
    calendar question regressed. Weighting by IDF, as the coverage measure
    itself does, makes a common identifier contribute almost nothing and a
    rare one contribute like the strong query term it is.

    Identifiers the index already holds are deliberately skipped. CSBS, NAAC
    and SGPA are ordinary index terms, so coverage can already see them;
    boosting them again only re-ranks pages that all carry the acronym, and
    measured over the 62 questions it traded one hit for another (r04 won,
    c01 lost) for no net gain. What is left is what the tokenizer destroyed -
    single letters, which are rare by construction and take the maximum IDF.
    """
    mx = max(idf.values()) if idf else 1.0
    return [(rx, mx) for rx, label in ids if label.lower() not in idf]


def id_bonus(text, ids_w, den):
    """IDF share of the query's identifiers present, with case, in this chunk.
    `den` is the coverage denominator, so the bonus is on coverage's scale."""
    if not ids_w or den <= 0:
        return 0.0
    return sum(w for rx, w in ids_w if rx.search(text)) / den


W["ident"] = 0.6                        # weight of identifier coverage, like W["head"]
W["toc"] = 0.4
W["url"] = 0.6                          # query term found in the document's URL/file name
W["glossary"] = 0.6                     # definition question x chunk that is a glossary naming the term
W["site_top"] = 1.2                     # /about, /placements, /academics: canonical overview pages
W["enum"] = float(os.environ.get("VJLM_ENUM", "0"))    # rejected: 83% at 0.5, 81% at 1.0
W["branch_match"] = float(os.environ.get("VJLM_BRANCH_MATCH", "1.5"))
W["branch_mismatch"] = float(os.environ.get("VJLM_BRANCH_MISMATCH", "0.35"))
W["hub"] = float(os.environ.get("VJLM_HUB", "0"))     # rejected: see SENT_W in rag/retrieve.py
W["bm25_tie"] = 0.2                     # tie-break between chunks that cover the same terms
W["per_doc"] = {"regulation": 3, "scheme": 1, "site": 2}   # long rulebooks answer many different questions


# ----------------------------------------------------------- word forms
def variants(t):
    """Inflected forms of a query term, scored as the SAME word (full credit).
    The index has no stemming, so 'requirement' did not match 'requirements'
    and 'conducted' did not match 'conduct'. Same lemma, not a synonym."""
    v = set()
    if t.endswith("ies") and len(t) > 4:
        v.add(t[:-3] + "y")
    if t.endswith("es") and len(t) > 3:
        v.add(t[:-2]); v.add(t[:-1])
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        v.add(t[:-1])
    if not t.endswith("s"):
        v.add(t + "s"); v.add(t + "es")
        if t.endswith("y") and len(t) > 3:
            v.add(t[:-1] + "ies")
    if t.endswith("ing") and len(t) > 5:
        v.add(t[:-3]); v.add(t[:-3] + "e")
    if t.endswith("ed") and len(t) > 4:
        v.add(t[:-2]); v.add(t[:-1])
    v.discard(t)
    return v


def cover_score(tf, terms, w_syn):
    """IDF-weighted coverage over (term, idf, synonyms, variants) tuples:
    1.0 term or an inflection of it, w_syn for a synonym, else 0."""
    num = den = 0.0
    for t, idf, syns, var in terms:
        den += idf
        if t in tf or any(x in tf for x in var):
            num += idf
        elif syns:
            for s in syns:
                if s in tf:
                    num += w_syn * idf
                    break
    return num / den if den else 0.0


# A question asking for a SET rather than a fact: which clubs, what are the
# bodies, how many branches, list the labs.
_LISTY = re.compile(r"^\s*(which|what are|list|name the)\b|\bhow many\b|"
                    r"\ball (?:the )?\w+s\b", re.I)


def enum_bonus(tf, terms, den):
    """How INSISTENTLY a chunk repeats the question's own words, IDF-weighted.

    Coverage measures BREADTH - each term counts once, however often it occurs -
    and that is what makes it robust. But an enumeration page answers a "which
    X" question precisely by repeating X: /campus-life/clubs says "club" a dozen
    times while listing them, and shares almost nothing else with "which
    societies and clubs can I join on campus". It was ranked below forty
    individual club pages, each of which says "club" once or twice
    (evaluation/miss_report.py, k01). BM25 does use term frequency, but it
    normalises by document length, which is exactly backwards for a long list,
    and it enters the score only as a 0.2-weighted tie-break.
    """
    if not den:
        return 0.0
    s = 0.0
    for t, idf, syns, var in terms:
        c = tf.get(t, 0)
        for v in var:
            c = max(c, tf.get(v, 0))
        if c > 1:
            s += (idf / den) * min(1.0, math.log1p(c - 1) / math.log(9.0))
    return min(1.0, s)


_UNIT = re.compile(r"(?<=[.;:!?])\s+|\n+")


def best_unit(text, terms, den, w_syn=0.5):
    """Max coverage of the question by any ONE sentence, as a share of its IDF.

    All three arms read the whole chunk: coverage counts term frequencies over
    1200 characters and the embedding averages them. So a long page that
    mentions the question's words in five unrelated places scores like a clause
    whose single sentence states the answer, and that is where the remaining
    misses sit - the gold clause at rank 6-26 under pages that are merely about
    the topic (evaluation/miss_report.py). This asks the sharper question: is
    there one sentence here that covers the question?

    Pure Python over a few dozen candidates, so it adds no model and no memory.
    """
    if not den:
        return 0.0
    from rag.index import tok
    best = 0.0
    for u in _UNIT.split(text):
        if len(u) < 16:
            continue
        tf = set(tok(u))
        if not tf:
            continue
        s = 0.0
        for t, idf, syns, var in terms:
            if t in tf or any(x in tf for x in var):
                s += idf
            elif syns and any(x in tf for x in syns):
                s += w_syn * idf
        if s > best:
            best = s
    return best / den


def is_glossary(text):
    return bool(_GLOSSARY.search(text[:400])) or len(_GLOSS_ROW.findall(text)) >= 5


def glossary_bonus(text, ids, den_unused=None):
    """1.0 when the chunk is a glossary/abbreviation table that contains one of
    the question's identifiers (case intact), else 0."""
    if not is_glossary(text):
        return 0.0
    return 1.0 if any(rx.search(text) for rx, _ in ids) else 0.0


def cover_parts(tf, terms):
    """-> (keyword share, synonym-only share) of the query's IDF mass.

    Same walk as cover_score, but kept apart so each can carry its own weight:
    cover_score == keyword + w_syn * synonym."""
    kw = sy = den = 0.0
    for t, idf, syns, var in terms:
        den += idf
        if t in tf or any(x in tf for x in var):
            kw += idf
        elif syns and any(x in tf for x in syns):
            sy += idf
    return (kw / den, sy / den) if den else (0.0, 0.0)
