"""Structure-aware chunking: every chunk is a whole thought that names where it is.

Why the old chunker made retrieval hard
---------------------------------------
extract.chunk() packed lines into 1,100-character windows page by page. That cut
a rule away from its heading ("15.2.1 ... 35% marks in the SEE" sat apart from
"15.0 MINIMUM ACADEMIC REQUIREMENTS"), split sections at every page break, began
chunks mid-sentence, and left page numbers and running headers inside the text.
A student's question is phrased around the heading ("what do I need to pass a
course?"); the clause that answers it rarely repeats the heading's words, so
neither keyword matching, synonyms nor embeddings could connect them.

What this does instead
----------------------
1. Reads a document as ONE stream (pages only remember where a unit started), so a
   section that runs over a page break stays whole.
2. Drops what is not content: bare page numbers, and short lines that repeat on
   many pages (running headers/footers). Re-joins hard-wrapped lines.
3. Cuts the stream into logical UNITS - a numbered clause, a bullet, a heading, a
   paragraph - and never splits a unit unless it alone exceeds the size limit,
   and then only at a sentence end.
4. Tracks the heading path (document > 15.0 MINIMUM ACADEMIC REQUIREMENTS >
   15.6 Detention ...) and packs units into chunks without crossing a top-level
   section.
5. Returns the heading path as `context`, SEPARATE from `text`. Retrieval indexes
   context + text (so the heading's words, and its meaning for the embedder, are
   part of every clause under it); the model is shown `text` only, because the
   fine-tuned model copies the first sentence it is given and must not copy a
   breadcrumb.
6. Web pages (already one element per line) get the same treatment: runs of short
   lines - tables and statistic tiles that the renderer flattened one cell per
   line - are joined so a row's values stay next to each other.
"""
import re
from collections import Counter

MAX_CHARS = 1200        # hard ceiling for a chunk
TARGET = 950            # start a new chunk at the next unit boundary past this
MIN_CHARS = 120         # a smaller tail is merged into its neighbour
WRAP_MIN = 62           # a PDF line this long that does not end a sentence continues on the next

_PAGENO = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?\s*$", re.I)
_TOP = re.compile(r"^\s*(\d{1,2})\.0\s+([A-Z][A-Za-z0-9 ,&/\-'()]{3,110}?)\s*:?\s*$")
_SUB = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s+([A-Z][A-Za-z0-9 ,&/\-'()]{3,90}?)\s*:\s*$")
_CLAUSE = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,3}){1,4}|\d{1,3}[.)]|[a-z][.)]|[ivx]{1,5}[.)]|[•●▪◦■\-–*])\s+\S")
_LISTITEM = re.compile(r"^\s*(?:[a-z][.)]|[ivx]{1,5}[.)])\s+\S")
# a table row: a course code, or three or more trailing numbers (L T P CH C)
_ROWLIKE = re.compile(r"\b\d{2}[A-Z]{2}\d[A-Z]{2}\d{3}\b|(?:\s\d{1,3}){3,}\s*$")
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&/\-:'()]{5,90}$")
_SENT_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(•])")
_PUNCT_END = re.compile(r"[.!?:;)\]”\"]\s*$")


def _recurring(pages, share=0.3):
    """Short lines found on many pages of one document: running headers/footers."""
    if len(pages) < 4:
        return set()
    seen = Counter()
    for _, t in pages:
        seen.update({l.strip() for l in t.split("\n") if l.strip()})
    cut = max(3, int(len(pages) * share))
    return {l for l, n in seen.items() if n >= cut and len(l) < 100}


def _lines(pages, html):
    """-> [(page, line)] with blank/noise lines removed."""
    drop = set() if html else _recurring(pages)
    out = []
    for pno, t in pages:
        for raw in t.split("\n"):
            l = raw.strip()
            if not l or l in drop:
                continue
            if not html and _PAGENO.match(l):
                continue
            out.append((pno, l))
    return out


def _is_heading(l):
    return bool(_TOP.match(l) or _SUB.match(l)
                or (_ALLCAPS.match(l) and not _PAGENO.match(l)))


def _units(lines, html):
    """-> [(page, text, kind)] kind in {'top','sub','head','body'}."""
    units, cur, cur_page = [], [], None

    def flush():
        nonlocal cur, cur_page
        if cur:
            units.append((cur_page, " ".join(cur), "body"))
        cur, cur_page = [], None

    short_run = []

    def flush_run():
        # a flattened table / stat tiles: one cell per line, kept exactly as
        # rendered so "60 / Seats / Intake" and "Dr. X / Dean" stay readable
        nonlocal short_run
        for p, x in short_run:
            units.append((p, x, "body"))
        short_run = []

    if html:
        marked = []
        for i, (pno, l) in enumerate(lines):
            nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
            prv = lines[i - 1][1] if i else ""
            if (3 < len(l) <= 60 and l[0].isupper() and not _PUNCT_END.search(l)
                    and len(nxt) >= 60 and (not prv or len(prv) >= 45 or _PUNCT_END.search(prv))):
                marked.append((pno, l, True))
            else:
                marked.append((pno, l, False))
    else:
        marked = [(p, l, False) for p, l in lines]

    for pno, l, is_title in marked:
        if is_title:
            flush(); flush_run()
            units.append((pno, l, "sub"))
            continue
        m_top, m_sub = _TOP.match(l), _SUB.match(l)
        if m_top or m_sub or (_ALLCAPS.match(l) and len(l) < 90):
            flush(); flush_run()
            units.append((pno, l, "top" if m_top else "sub" if m_sub else "head"))
            continue
        if html:
            if len(l) < 45 and not _PUNCT_END.search(l):
                flush(); short_run.append((pno, l)); continue
            flush_run()
            units.append((pno, l, "body"))
            continue
        if _ROWLIKE.search(l):
            flush(); units.append((pno, l, "body"))     # a table row stays one line
            continue
        if _CLAUSE.match(l):
            flush(); cur, cur_page = [l], pno
        elif cur and len(cur[-1]) >= WRAP_MIN and not _PUNCT_END.search(cur[-1]):
            cur.append(l)               # hard-wrapped continuation: previous line ran full width
        else:
            flush(); cur, cur_page = [l], pno   # a short line ended its paragraph / is a table row
    flush(); flush_run()
    return units


def _split_long(text):
    """Break an over-long unit at sentence ends; last resort, at a space."""
    parts, cur = [], ""
    for s in _SENT_END.split(text):
        if len(cur) + len(s) + 1 <= MAX_CHARS:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                parts.append(cur)
            while len(s) > MAX_CHARS:
                cut = s.rfind(" ", 0, MAX_CHARS)
                cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
                parts.append(s[:cut].strip()); s = s[cut:].strip()
            cur = s
    if cur:
        parts.append(cur)
    return parts


def chunk_document(pages, title="", html=False):
    """pages: [(page_no, text)] -> [{"page", "context", "text"}].

    `context` is "title > top section > sub-section"; `text` is the body only."""
    units = _units(_lines(pages, html), html)
    top = sub = ""
    chunks, buf, buf_page, buf_ctx = [], [], None, ""
    leadin = ""         # last "...as follows:" unit, kept with the list items it introduces

    def ctx():
        return " > ".join(x for x in (title, top, sub) if x)

    def flush():
        nonlocal buf, buf_page
        if buf:
            chunks.append({"page": buf_page, "context": buf_ctx, "text": "\n".join(buf)})
        buf, buf_page = [], None

    for pno, text, kind in units:
        if kind == "top":
            flush(); top, sub = text.rstrip(":").strip(), ""
            continue
        if kind == "sub":
            if sum(map(len, buf)) >= MIN_CHARS:
                flush()
            sub = text.rstrip(":").strip()
            continue
        if kind == "head":
            if sum(map(len, buf)) >= MIN_CHARS:
                flush()
            top, sub = text.strip(), ""
            continue
        pieces = _split_long(text) if len(text) > MAX_CHARS else [text]
        for piece in pieces:
            size = sum(map(len, buf)) + len(buf)
            if buf and (size + len(piece) > MAX_CHARS or (size >= TARGET)):
                flush()
            if not buf:
                buf_page, buf_ctx = pno, ctx()
                if leadin and _LISTITEM.match(piece):
                    buf.append(leadin)          # a list never starts without its lead-in
            buf.append(piece)
            leadin = piece if piece.rstrip().endswith(":") else (leadin if _LISTITEM.match(piece) else "")
    flush()

    # fold a tiny tail into the chunk before it when they share a context
    merged = []
    for c in chunks:
        if (merged and len(c["text"]) < MIN_CHARS and merged[-1]["context"] == c["context"]
                and len(merged[-1]["text"]) + len(c["text"]) <= MAX_CHARS + 200):
            merged[-1]["text"] += "\n" + c["text"]
        else:
            merged.append(c)
    return [c for c in merged if len(c["text"]) >= 40]


def doc_title(url, pages, html):
    """A short human title for the breadcrumb: the page's own title for web pages,
    the file name for documents (R25 and BTECH in a file name are real context)."""
    from urllib.parse import unquote
    if html:
        first = next((l.strip() for _, t in pages for l in t.split("\n") if l.strip()), "")
        return re.sub(r"\s*[|\-\u2013]\s*VNRVJIET.*$", "", first).strip()[:90]
    stem = unquote(url.rstrip("/").split("/")[-1])
    stem = re.sub(r"\.(pdf|docx|xlsx)$", "", stem, flags=re.I)
    return re.sub(r"[-_]+", " ", stem).strip()[:90]
