"""Stage 1 of the dynamic pipeline: discovery.

Re-runnable. Finds every page route and document URL the site publishes, by
combining sitemap.xml with the route/asset table baked into the app bundle.
Stdlib only. No third-party service, no LLM.
"""
import csv, json, os, re, sys, urllib.request, urllib.error, gzip, io, time

WWW = "https://www.vnrvjiet.ac.in"
UA = "VJLM-research-crawler (VNRVJIET academic project)"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data_catalog")

# robots.txt on www: Disallow /api/ and /nirf2026patent. docs.* disallows all.
BLOCKED_HOSTS = {"docs.vnrvjiet.ac.in"}
BLOCKED_PATHS = ("/api/", "/nirf2026patent")

CATEGORIES = [
    ("academic_regulations", r"regulation|attendance|detention|grading|cgpa|\br\d{2}\b|\ba\d{2}\b"),
    ("academic_calendar",    r"calender|calendar|almanac|commencement"),
    ("syllabus",             r"syllab|curricul|course-structure|scheme|meme|credit|elective"),
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
    if any(h in url for h in BLOCKED_HOSTS):
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

    # routes referenced by the router but missing from the sitemap
    for r in re.findall(r'"(/(?:about|admissions|academics|departments|placements|research|'
                        r'campus-life|alumni|careers|events|press|contact)[a-zA-Z0-9/-]*)"', bundle):
        routes.add(WWW + r)

    assets = set()
    for a in re.findall(r'"(/__l5e/assets-v1/[^"]+|https://[a-z0-9.-]*vnrvjiet\.ac\.in/[^"]+)"', bundle):
        assets.add(a if a.startswith("http") else WWW + a)

    docs = sorted(u for u in assets if u.lower().endswith(".pdf"))
    print(f"      {len(routes)} routes total, {len(docs)} PDFs total")

    print("[3/3] writing catalog ...", flush=True)
    rows, skipped = [], 0
    for url in sorted(routes) + docs:
        if not allowed(url):
            skipped += 1
            continue
        path = url.split("vnrvjiet.ac.in", 1)[-1]
        kind = "pdf" if url.lower().endswith(".pdf") else "html"
        # for PDFs the uuid folder is noise; classify on the filename
        basis = path.rsplit("/", 1)[-1] if kind == "pdf" else path
        cat = classify(basis)
        sid = re.sub(r"[^a-z0-9]+", "-", (kind + basis.lower()))[:90].strip("-")
        rows.append({
            "source_id": sid,
            "canonical_url": url,
            "category": cat,
            "document_type": kind,
            "audience": "student|faculty",
            "visibility": "PUBLIC",
            "update_frequency": FREQ.get(cat, "monthly"),
            "parser_type": "pdf_text" if kind == "pdf" else "rendered_dom",
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
    print(f"  skipped {skipped} (robots.txt disallowed)")
    print(f"  types: {dict(k)}")
    print("  categories:")
    for name, n in c.most_common():
        print(f"    {n:5d}  {name}")

if __name__ == "__main__":
    main()
