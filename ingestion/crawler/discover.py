"""Stage 1 of the dynamic pipeline: discovery.

Re-runnable. Finds every page route and document URL the site publishes, by
combining sitemap.xml with the route/asset table baked into the app bundle.
Stdlib only. No third-party service, no LLM.
"""
import csv, json, os, re, sys, urllib.request, urllib.error, gzip, io, time, hashlib
from urllib.parse import unquote

WWW = "https://www.vnrvjiet.ac.in"
UA = "VJLM-research-crawler (VNRVJIET academic project)"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data_catalog")

# robots.txt: www disallows /api/ and /nirf2026patent; docs.* disallows /.
#
# INGEST-002, updated 2026-09-30: the project extracts everything the official
# site publishes. VNRVJIET's IT team reported that robots.txt on docs.* is not
# in their control and cannot be changed, and the rulebooks, condonation forms
# and older calendars students need exist only there. The rules are recorded
# here but not applied to content the official site itself links to:
#   * docs.vnrvjiet.ac.in - DOCUMENTS ONLY, and only URLs linked from www (its
#     bundle or a rendered page): what a student gets by pressing "Download
#     PDF". The host is never crawled, listed or walked.
#   * /nirf2026patent - a page www itself links to.
# Still excluded: /api/ (data endpoints, not pages) and /admin (CMS login).
# The switch lives in configs/ingestion.yaml (scope.respect_robots_txt), the
# file someone would actually edit. Read with a regex so this stays stdlib-only;
# if the key cannot be read, fall back to strict compliance - the safe side.
def _respect_robots():
    try:
        with open(os.path.join(ROOT, "configs", "ingestion.yaml"), encoding="utf-8") as f:
            m = re.search(r"^\s*respect_robots_txt:\s*(true|false)\b", f.read(), re.M)
        return m is None or m.group(1) == "true"
    except OSError:
        return True
ROBOTS_OVERRIDE = not _respect_robots()
LINKED_DOCUMENT_HOSTS = {"docs.vnrvjiet.ac.in"}
BLOCKED_PATHS = ("/api/",)
# Not blocked by robots.txt, but not knowledge: the CMS login.
EXCLUDED_ROUTES = ("/admin",)

# Document types the pipeline can read. Everything else linked (photos, video)
# carries no extractable text.
DOC_EXT = (".pdf", ".docx", ".xlsx")
LINKS = os.path.join(ROOT, "ingestion", "state", "rendered_links.json")

# First match wins, so order is policy. Calendars go first because their
# filenames carry the regulation code too ("...-Calendar-2026-2027-R25.pdf").
# A bare regulation code is NOT evidence of a rulebook: every syllabus booklet
# is named "II-B.Tech-CSE-R25.pdf". Matching \br\d{2}\b here filed all 32 of
# them - 5,269 chunks - as academic_regulations, so a credits table was cited
# to students as a regulation (CORPUS-001). A code alone now means syllabus.
CATEGORIES = [
    ("academic_calendar",    r"calender|calendar|almanac|commencement"),
    # Rulebooks named by code alone: VNRA19, VNRR19, VNRR18 (B.Tech) and
    # VNRMTR25, VNRMTR22, VNRMTA18, VNRMT18 (M.Tech - linked only behind the
    # M.Tech button on /academics/regulations, found by render.py's clicks).
    ("academic_regulations", r"regulation|attendance|detention|grading|cgpa|\bvnr(?:mt)?[ar]?\d{2}\b"),
    ("syllabus",             r"syllab|curricul|course-structure|scheme|meme|credit|elective|\b[ra]\d{2}\b"),
    ("examinations",         r"exam|timetable|time-table|question-paper|malpractice|revaluation"),
    ("placements",           r"placement|recruit|internship|ctc"),
    ("faculty",              r"\bcv\b|-cv$|profile|publication|^dr-|-ka-\d{4}"),
    ("departments",          r"department|dept|nba|naac|nirf|peo|pso|newsletter|annual-report"),
    ("fees_scholarships",    r"\bfee\b|fees|scholarship|tuition|refund|dues"),
    ("forms_procedures",     r"form|condonation|no-dues|affidavit|undertaking|application"),
    ("admissions",           r"admission|eamcet|ecet|pgecet|counsel|intake"),
    ("clubs_events",         r"club|fest|event|convergence|sports|cultural|nss"),
    ("research",             r"research|patent|idea-lab|incubat|consultanc|journal|conference"),
    ("governance_grievance", r"grievance|anti-ragging|\bicc\b|iqac|committee|minutes|policy|rti|code-of-conduct"),
    ("facilities",           r"hostel|library|transport|canteen|facilit|infrastructure|health"),
    ("about",                r"about|governance|accreditation|csr|vignana-jyothi|history"),
]

# Which categories are time-sensitive -> how often to recheck.
FREQ = {
    "examinations": "daily", "placements": "weekly", "clubs_events": "weekly",
    "academic_calendar": "weekly", "admissions": "weekly", "faculty": "monthly",
    "departments": "monthly", "fees_scholarships": "monthly", "research": "monthly",
    "forms_procedures": "monthly", "governance_grievance": "monthly",
    "facilities": "yearly", "about": "yearly", "syllabus": "yearly",
    "academic_regulations": "yearly",
}

def fetch(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", "replace")

def classify(text):
    t = text.lower()
    for name, pat in CATEGORIES:
        if re.search(pat, t):
            return name
    return "general"

def allowed(url):
    host = re.match(r"https?://([^/]+)", url).group(1).lower()
    if host in LINKED_DOCUMENT_HOSTS:
        if not ROBOTS_OVERRIDE or not url.lower().endswith(DOC_EXT):
            return False
    if not ROBOTS_OVERRIDE and "/nirf2026patent" in url:
        return False
    return not any(p in url for p in BLOCKED_PATHS)

def main():
    print("[1/3] sitemap ...", flush=True)
    sm = fetch(f"{WWW}/sitemap.xml")
    routes = set(re.findall(r"<loc>([^<]+)</loc>", sm))
    print(f"      {len(routes)} routes from sitemap")

    print("[2/3] app bundle ...", flush=True)
    shell = fetch(WWW + "/")
    m = re.search(r'src="(/assets/index-[^"]+\.js)"', shell)
    if not m:
        print("!! bundle script tag not found - site structure changed"); sys.exit(1)
    bundle_url = WWW + m.group(1)
    print(f"      {bundle_url}")
    bundle = fetch(bundle_url, timeout=180)
    print(f"      {len(bundle):,} bytes")

    # Section routes referenced anywhere as a quoted string - including data
    # fields like the club list's "/campus-life/clubs/isie", which are neither
    # router paths nor href/to attributes. Dropping this pattern once retired
    # 8 club pages, so it stays alongside the two below.
    for r in re.findall(r'"(/(?:about|admissions|academics|departments|placements|research|'
                        r'campus-life|alumni|careers|events|press|contact)[a-zA-Z0-9/-]*)"', bundle):
        routes.add(WWW + r)

    # Routes the app itself declares or links to. The sitemap is a subset:
    # /page/examination-cell, /rti, /ugc-compliance and others are routed and
    # linked but not listed. Templates (":slug") are skipped here; their real
    # instances only exist in rendered pages and arrive via LINKS below.
    for r in re.findall(r'\bpath:"(/[a-z][^"]*)"', bundle) + \
             re.findall(r'(?:href|to):"(/[a-z][^"#?]*)"', bundle):
        if ":" not in r and "*" not in r and not r.lower().endswith(DOC_EXT) \
                and not r.startswith(("/__l5e", "/assets", "/images", "/~")):
            routes.add(WWW + r.rstrip("/"))

    assets = set()
    # Documents are linked three ways: CMS assets, absolute URLs, and plain
    # root-relative paths (/docs/cse/Placements-2025-26-AY.pdf). Matching only
    # the first two missed 24 PDFs on www. Markdown content blocks use
    # "](/path)" rather than quotes, so both forms are read.
    # Filenames on docs.* contain spaces ("civildept/3 Alumni Scholarship.pdf"),
    # so a URL runs to the closing quote, not to the first whitespace. URLs are
    # stored decoded; the browser's percent-encoded spelling of the same file
    # (from rendered-page links) must not become a second source.
    for a in re.findall(r'"(/[^"\n]{1,300}?|https://[a-z0-9.-]*vnrvjiet\.ac\.in/[^"\n]{1,300}?)"', bundle) + \
             re.findall(r'\]\((https://[a-z0-9.-]*vnrvjiet\.ac\.in/[^)\n]{1,300}|/[^)\n]{1,300})\)', bundle):
        if a.lower().endswith(DOC_EXT):
            assets.add(unquote(a if a.startswith("http") else WWW + a))

    # What the rendered pages link to (render.py writes it). This is how
    # template-built pages - faculty profiles, blog posts - are found at all.
    harvested = 0
    if os.path.exists(LINKS):
        with open(LINKS, encoding="utf-8") as f:
            for u in json.load(f):
                d = re.match(r"https?://[a-z0-9.-]*vnrvjiet\.ac\.in/[^?#]*", u)
                if d and not re.match(r"https?://(?:www\.)?vnrvjiet\.ac\.in/", u) \
                        and d.group(0).lower().endswith(DOC_EXT):
                    n = len(assets); assets.add(unquote(d.group(0))); harvested += len(assets) - n
                    continue
                m = re.match(r"https?://(?:www\.)?vnrvjiet\.ac\.in(/[^?#]*)", u)
                if not m:
                    continue
                path = m.group(1).rstrip("/") or ""
                if path.lower().endswith(DOC_EXT):
                    n = len(assets); assets.add(unquote(WWW + path)); harvested += len(assets) - n
                elif not path.startswith(("/__l5e", "/assets", "/images", "/~")) \
                        and "." not in path.rsplit("/", 1)[-1]:
                    n = len(routes); routes.add(WWW + path); harvested += len(routes) - n
    routes = {r for r in routes if not any(r.split("vnrvjiet.ac.in", 1)[-1].startswith(x)
                                           for x in EXCLUDED_ROUTES)}
    # One spelling per page: "/about/" and "/about" are the same source.
    routes = {(r.rstrip("/") or r) if r.rstrip("/") != WWW else WWW + "/" for r in routes}
    print(f"      {harvested} new URLs from rendered-page links")

    docs = sorted(assets, key=lambda u: (not u.startswith(WWW), u))
    print(f"      {len(routes)} routes total, {len(docs)} documents total")

    print("[3/3] writing catalog ...", flush=True)
    rows, skipped, seen, seen_urls = [], 0, set(), set()
    for url in sorted(routes) + docs:
        if not allowed(url):
            skipped += 1
            continue
        path = url.split("vnrvjiet.ac.in", 1)[-1]
        ext = url.lower().rsplit(".", 1)[-1]
        kind = ext if url.lower().endswith(DOC_EXT) else "html"
        # for PDFs the uuid folder is noise; classify on the filename
        basis = path.rsplit("/", 1)[-1] if kind != "html" else path
        cat = classify(basis)
        if cat == "general" and kind != "html":
            # The filename is the primary evidence; the folders are the
            # fallback. On docs.* they are often the only signal
            # (/pdfs/regulations/R25/..., /pdfs/syllabus/...), while www's
            # uuid folders carry none and change nothing.
            cat = classify("/".join(path.split("/")[:-1]))
        sid = re.sub(r"[^a-z0-9]+", "-", (kind + basis.lower()))[:90].strip("-")
        # The id is built from the filename alone, which is not unique across
        # folders ("civildept/1.pdf", "eeedept/1.pdf"). A second, different
        # file with the same name gets a URL-hash suffix instead of being
        # dropped; the same URL seen twice is skipped.
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if sid in seen:
            sid = sid[:80] + "-" + hashlib.sha1(url.encode()).hexdigest()[:8]
        seen.add(sid)
        rows.append({
            "source_id": sid,
            "canonical_url": url,
            "category": cat,
            "document_type": kind,
            "audience": "student|faculty",
            "visibility": "PUBLIC",
            "update_frequency": FREQ.get(cat, "monthly"),
            "parser_type": {"pdf": "pdf_text", "html": "rendered_dom"}.get(kind, "office_xml"),
            "fine_tune_usage": "no",          # set per Phase 2 classification
            "rag_usage": "full_source",
            "active": "true",
        })

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "catalog.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    c = Counter(r["category"] for r in rows)
    k = Counter(r["document_type"] for r in rows)
    print(f"\n  wrote {len(rows)} sources -> {out}")
    print(f"  skipped {skipped} (excluded: /api/, /admin, or ROBOTS_OVERRIDE off)")
    print(f"  types: {dict(k)}")
    print("  categories:")
    for name, n in c.most_common():
        print(f"    {n:5d}  {name}")

if __name__ == "__main__":
    main()
