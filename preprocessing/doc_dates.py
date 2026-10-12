"""Document dates, so the newest source wins when sources disagree.

Why this exists
---------------
The corpus contradicts itself and nothing in a chunk said which statement was
current. Measured case: a student asking which cards are refused for online
fee payment is answerable from two PDFs,

  VNR-Online-Fee-Payment-Website-Guidelines.pdf  "Rupay Credit & Rupay Debit
      Cards are not allowed to pay the fee."          created 2024-08-06
  VNR-VJIET-Online-Fee-payment-terms-conditions.pdf  "The Institute accepts
      payments through ... RuPay Cards"                created 2026-05-14

Both are public, both are authentic, and the older one is wrong today. The
crawl state cannot settle it - its timestamps say when WE fetched a file, not
when the college wrote it - so the date is read from the document itself:
/CreationDate and /ModDate for PDFs, the core properties for .docx.

Regulations are deliberately NOT resolved this way. R22 and R25 coexist on
purpose, a student may be governed by either, and rag/scope.py already picks
between them by regulation code; recency would wrongly bury R22 for the
students it applies to.

    python preprocessing/doc_dates.py        -> preprocessing/out/doc_dates.json
"""
import json
import os
import re
import sys
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "ingestion", "snapshots")
OUT = os.path.join(ROOT, "preprocessing", "out", "doc_dates.json")

# D:20260514154356+05'30'
_PDFDATE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")
_XMLDATE = re.compile(r"<dcterms:(?:modified|created)[^>]*>(\d{4}-\d{2}-\d{2})")


def _iso(y, m, d):
    try:
        return datetime(int(y), int(m), int(d)).date().isoformat()
    except ValueError:
        return None


def pdf_date(path):
    """Newest of /ModDate and /CreationDate, read without a full parse."""
    try:
        from pypdf import PdfReader
    except ImportError:                              # older environments
        from PyPDF2 import PdfReader                 # noqa: N813
    try:
        meta = PdfReader(path).metadata or {}
    except Exception:
        return None
    found = []
    for key in ("/ModDate", "/CreationDate"):
        m = _PDFDATE.search(str(meta.get(key, "")))
        if m:
            got = _iso(*m.groups())
            if got:
                found.append(got)
    return max(found) if found else None


def docx_date(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("docProps/core.xml").decode("utf-8", "replace")
    except Exception:
        return None
    dates = _XMLDATE.findall(xml)
    return max(dates) if dates else None


def main():
    if not os.path.isdir(SNAP):
        print("no snapshots at", SNAP)
        return 1
    out, skipped = {}, 0
    names = sorted(os.listdir(SNAP))
    for n, name in enumerate(names, 1):
        path = os.path.join(SNAP, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if low.endswith(".pdf"):
            got = pdf_date(path)
        elif low.endswith((".docx", ".pptx", ".xlsx")):
            got = docx_date(path)
        else:
            continue
        # snapshot name is the source_id plus the original extension
        source_id = os.path.splitext(name)[0]
        if got:
            out[source_id] = got
        else:
            skipped += 1
        if n % 250 == 0:
            print("  %d/%d  dated=%d" % (n, len(names), len(out)), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    os.replace(tmp, OUT)
    print("dated %d documents, %d without a usable date -> %s" % (len(out), skipped, OUT))
    if out:
        ys = sorted(v[:4] for v in out.values())
        print("oldest %s  newest %s" % (ys[0], ys[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
