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


def doc_profile(row):
    """-> (family, regulation|None, programme|None, depth) from URL + chunk metadata.
    depth = number of path segments for site pages (/about = 1, /campus-life/clubs/x = 3)."""
    fam, reg, prog = _doc_profile(row)
    depth = 0
    if fam == "site":
        depth = len([x for x in re.sub(r"^https?://[^/]+", "", row.get("url") or "").split("/") if x])
    return fam, reg, prog, depth


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
    fam, reg, prog, depth = profile
    f = 1.0
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


def content_terms(tokens):
    """Query tokens minus document-identity words (kept if nothing else is left)."""
    kept = [t for t in tokens if t not in ID_TERMS]
    return kept if kept else tokens


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
W["site_top"] = 1.2                     # /about, /placements, /academics: canonical overview pages
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
