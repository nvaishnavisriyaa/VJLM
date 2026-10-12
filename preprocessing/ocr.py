"""OCR fallback for image-only PDFs.

Found 2026-09-30: 25 of 672 PDFs on www have no text layer - including four of
the five current academic calendars. pypdf returns nothing for them, extract.py
marked them 'empty', and nothing downstream read them. So the calendar existed
on the site, was fetched every refresh, and was never answerable.

Local and CPU-only: pages are rendered with pypdfium2 and read by RapidOCR's
ONNX build, which runs on the same onnxruntime fastembed already uses. No
Tesseract binary, no torch, no API; the models ship inside the wheel.

Two properties the unattended refresh depends on:
  * Results are cached by the SHA-256 of the PDF bytes. OCR costs ~20 s a page
    on CPU; an unchanged scanned PDF must never be read twice.
  * If the OCR stack is not installed this returns None and the document stays
    'empty', exactly as before. OCR is an addition, never a new way to fail.
"""
import hashlib, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "ingestion", "state", "ocr_cache")
DPI = 200            # below ~150 small table text starts to drop out
MAX_PAGES = 30       # a 300-page scan would stall the refresh for over an hour
ROW_TOL = 14         # px at DPI: boxes this close vertically share a table row

_engine = None


def _ocr_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        # use_cls=False: the text-orientation classifier took 85% of the time
        # (69 s vs 9.9 s a page, measured under load) and changed nothing -
        # identical 147 lines. It only helps with text printed upside down,
        # which scanned notices do not have; whole-page rotation is applied by
        # pdfium from the PDF itself.
        _engine = RapidOCR(use_cls=False)
    return _engine


# --- repairing the recogniser's output --------------------------------------
# The bundled model is Chinese+English and often drops the space between English
# words: "CommencementofClasses", "AcademicYear2026-2027". To BM25 each of those
# is one token no student will ever type, so the text is present but unfindable.
# Character-box gaps were measured and are too noisy to recover spaces from
# ("Program|10|mes" has the same gap as a real space). Every missing space seen
# was at a case change, a letter/digit boundary, or a glued function word, so
# those are repaired from the text itself.

_FULLWIDTH = str.maketrans("（）：，；［］", "():,;[]")
_GLUED = ("of", "for", "in", "the", "to", "and", "on", "at", "by", "with")
_ROMAN = {"Il": "II", "ll": "II", "lI": "II", "Ill": "III", "lll": "III", "l": "I"}


def _is_word(w):
    try:
        from nltk.corpus import wordnet as wn
        return bool(wn.synsets(w))
    except Exception:            # no WordNet -> skip that repair, keep the rest
        return None


def _unglue(tok):
    """'Marksin' -> 'Marks in', but leave 'within' alone: split only when the
    whole token is not a word and what remains is one."""
    low = tok.lower()
    if len(low) < 6 or not low.isalpha() or _is_word(low) is not False:
        return tok
    for fw in _GLUED:
        head = low[:-len(fw)]
        if low.endswith(fw) and len(head) >= 4 and _is_word(head):
            return tok[:-len(fw)] + " " + tok[-len(fw):]
    return tok


def respace(s):
    s = s.translate(_FULLWIDTH)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)             # CourseRegistration
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z]{2})", " ", s)      # CBCSProgrammes
    s = re.sub(r"(?<=[A-Za-z]{3})(?=\d)", " ", s)           # November2026 - not R25, PC1
    s = re.sub(r"(?<=\d)(?=[A-Za-z]{3})", " ", s)           # 2025Admitted - not 30th, 25PC
    s = re.sub(r"(?<=[)\]])(?=\w)|(?<=\w)(?=[(\[])", " ", s)
    s = re.sub(r"(?<=\.)(Il|ll|lI)(?=\s|$)", " II", s)       # B.Tech.Il -> B.Tech. II
    toks = [_ROMAN.get(t, t) for t in s.split()]
    return " ".join(_unglue(t) for t in toks)


def _rows(result):
    """Group OCR boxes into visual rows, left to right. Calendars and notices
    are tables; keeping a row on one line keeps 'Commencement of Classes' next
    to its date instead of 40 lines away from it."""
    items = sorted(result, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
    rows, cur, anchor = [], [], None
    for r in items:
        y = (r[0][0][1] + r[0][2][1]) / 2
        if cur and y - anchor > ROW_TOL:
            rows.append(cur)
            cur = []
        if not cur:
            anchor = y
        cur.append(r)
    if cur:
        rows.append(cur)
    return [" | ".join(respace(r[1]) for r in sorted(row, key=lambda r: r[0][0][0]))
            for row in rows]


def cache_path(path):
    with open(path, "rb") as f:
        return os.path.join(CACHE, hashlib.sha256(f.read()).hexdigest() + ".json")


def is_cached(path):
    """True if this exact file has been read before - costs a hash, not OCR."""
    return os.path.exists(cache_path(path))


def ocr_pdf(path):
    """[(page_no, text)] for an image-only PDF, or None if OCR is unavailable."""
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    cached = os.path.join(CACHE, digest + ".json")
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as f:
            return [tuple(p) for p in json.load(f)["pages"]]
    try:
        import numpy as np
        import pypdfium2 as pdfium
        engine = _ocr_engine()
    except Exception:
        return None

    pdf = pdfium.PdfDocument(path)
    pages = []
    for i in range(min(len(pdf), MAX_PAGES)):
        img = np.array(pdf[i].render(scale=DPI / 72).to_pil().convert("RGB"))
        result, _ = engine(img)
        pages.append((i + 1, "\n".join(_rows(result or []))))

    os.makedirs(CACHE, exist_ok=True)
    tmp = cached + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"pages": pages, "dpi": DPI, "total_pages": len(pdf),
                   "truncated": len(pdf) > MAX_PAGES}, f, ensure_ascii=False)
    os.replace(tmp, cached)
    return pages
