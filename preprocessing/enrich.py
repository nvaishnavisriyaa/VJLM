"""Layer 1 completion: metadata extraction, PII redaction, deduplication.

Design note: this deliberately does NOT rewrite chunk text in place. Embeddings
are computed over the original text, so mutating it would silently desync the
vectors from the metadata. Instead:

  * department / academic_year / regulation  -> added as metadata (no re-embed needed)
  * PII redaction                            -> applied at prompt-build time, so the
                                                model never sees emails or phone numbers
  * duplicates                               -> marked here, filtered at retrieval time

Run after extract.py. Rebuilding the index is not required.
"""
import hashlib, json, os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")
OUT = os.path.join(ROOT, "preprocessing", "out", "rag_chunks_enriched.jsonl")

DEPARTMENTS = {
    "cse-aiml": ["cse-aiml", "cse_aiml", "aiml", "artificial intelligence and machine learning"],
    "cse-ds":   ["cse-ds", "cse_ds", "data science"],
    "csbs":     ["csbs", "computer science and business systems"],
    "rai":      ["rai", "robotics and artificial intelligence"],
    "cse":      ["cse", "computer science and engineering"],
    "it":       ["information technology", "/it/", "-it-"],
    "ece":      ["ece", "electronics and communication"],
    "eee":      ["eee", "electrical and electronics"],
    "ei":       ["eie", "electronics and instrumentation"],
    "vlsi":     ["vlsi"],
    "mech":     ["mech", "mechanical engineering"],
    "civil":    ["civil engineering"],
    "ae":       ["automobile engineering", "aeronautical"],
    "biotech":  ["biotech", "biotechnology"],
    "chemistry": ["chemistry"],
    "physics":  ["physics"],
    "maths":    ["mathematics", "maths"],
    "english":  ["english", "humanities"],
    "mba":      ["mba", "management studies"],
    "mca":      ["mca"],
}

REGULATION = re.compile(r"\b([AR]\d{2})\b")
ACAD_YEAR = re.compile(r"\b(20\d{2})\s*[-–/]\s*(\d{2,4})\b")
YEAR = re.compile(r"\b(20[0-3]\d)\b")

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
LANDLINE = re.compile(r"(?<!\d)0?\d{2,4}[\s-]\d{6,8}(?!\d)")


def redact(text):
    """Strip direct contact PII. Used at prompt-build time, not at index time."""
    text = EMAIL.sub("[email redacted]", text)
    text = PHONE.sub("[phone redacted]", text)
    text = LANDLINE.sub("[phone redacted]", text)
    return text


# Departments are only inferred from body text for documents that ARE about a
# department. On a general page ("pay your B.Tech, M.Tech, MCA or Ph.D fees")
# a keyword in the first 400 characters says nothing about whose page it is -
# /payment was tagged "mca", and that tag reached the prompt and the sources
# panel (APP-002). Substring matching was also loose: "ece" matched "pieces".
DEPT_TEXT_CATEGORIES = {"departments", "faculty", "syllabus", "research"}
_DEPT_RX = {d: re.compile("|".join(r"(?<![a-z0-9])" + re.escape(k) + r"(?:(?![a-z0-9])|(?=dept))" for k in ks))
            for d, ks in DEPARTMENTS.items()}


def detect_department(url, text, category=None):
    hays = [url.lower()]
    if category in DEPT_TEXT_CATEGORIES:
        hays.append(text[:400].lower())
    for hay in hays:
        for dept, rx in _DEPT_RX.items():
            if rx.search(hay):
                return dept
    return None


def detect_year(url, text):
    hay = url + " " + text[:600]
    m = ACAD_YEAR.search(hay)
    if m:
        a, b = m.group(1), m.group(2)
        b = b if len(b) == 4 else a[:2] + b
        return "%s-%s" % (a, b[-2:])
    m = YEAR.search(hay)
    return m.group(1) if m else None


def norm_for_dedup(t):
    return hashlib.sha1(re.sub(r"[^a-z0-9]+", " ", t.lower()).strip().encode()).hexdigest()


def main():
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    print("loaded %d chunks" % len(rows))

    seen = {}
    dept_n = year_n = pii_n = dup_n = 0
    for r in rows:
        d = detect_department(r["url"], r["text"], r.get("category"))
        if d:
            r["department"] = d
            dept_n += 1
        y = detect_year(r["url"], r["text"])
        if y:
            r["academic_year"] = y
            year_n += 1
        m = REGULATION.search(r["url"] + " " + r["text"][:300])
        if m:
            r["regulation"] = m.group(1).upper()

        if EMAIL.search(r["text"]) or PHONE.search(r["text"]):
            r["has_pii"] = True
            pii_n += 1

        h = norm_for_dedup(r["text"])
        if h in seen:
            r["duplicate_of"] = seen[h]
            dup_n += 1
        else:
            seen[h] = r["chunk_id"]

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dc = Counter(r.get("department") for r in rows if r.get("department"))
    print("\ndepartment tagged : %d chunks across %d departments" % (dept_n, len(dc)))
    for d, n in dc.most_common(10):
        print("    %-10s %d" % (d, n))
    print("academic_year     : %d" % year_n)
    print("contains PII      : %d  (redacted at prompt-build time)" % pii_n)
    print("exact duplicates  : %d  (filtered at retrieval)" % dup_n)
    print("\nwrote -> %s" % OUT)


if __name__ == "__main__":
    main()
