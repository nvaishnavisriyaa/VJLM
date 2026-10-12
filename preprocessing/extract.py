"""Stage 3: extract text from snapshots -> normalized, chunked JSONL for RAG.

Structure-aware chunking with page provenance so every chunk can cite the
exact document and page it came from.
"""
import argparse, json, os, re, sys, csv, hashlib, time
from collections import Counter
from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "ingestion", "snapshots")
STATE = os.path.join(ROOT, "ingestion", "state", "sync_state.json")
CATALOG = os.path.join(ROOT, "data_catalog", "catalog.csv")
OUT = os.path.join(ROOT, "preprocessing", "out")
MIN_CHARS, MAX_CHARS, OVERLAP = 200, 1100, 150

# OCR budget per run, seconds. docs.* is 60% scans at ~7 pages each - about
# 5,500 pages, many hours of CPU. Done in one run, that would hold every web
# page and text PDF out of the knowledge base until it finished. So OCR gets a
# budget: scans beyond it are DEFERRED to a backlog, the run completes and is
# indexed, and every later run continues the backlog (refresh.py loops quickly
# while one exists). Cached scans cost nothing and never count against it.
OCR_BUDGET_S = float(os.environ.get("VJLM_OCR_BUDGET_S", "1200"))
OCR_BACKLOG = os.path.join(OUT, "ocr_backlog.json")
# {source_id: content_hash of the snapshot it was last extracted from}. Written
# for every source extraction finished with - text, empty, or unrendered shell -
# and never for a failure or a deferral, which must be tried again. It is what
# lets extraction, and refresh.py, see work that no cycle ever finished: a
# document fetched by a cycle that was then stopped used to look "unchanged"
# forever after and never reach the knowledge base (INGEST-007).
EXTRACTED = os.path.join(OUT, "extracted.json")
_ocr_spent = 0.0
sys.path.insert(0, ROOT)

def normalize(t):
    t = t.replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    t = re.sub(r"(?m)^\s*Page \d+ of \d+\s*$", "", t)
    return t.strip()

def chunk(text, max_c=MAX_CHARS, ov=OVERLAP):
    """Split on paragraph boundaries, pack up to max_c, overlap for context."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_c:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur: out.append(cur)
            if len(p) > max_c:
                for i in range(0, len(p), max_c - ov):
                    out.append(p[i:i + max_c])
                cur = ""
            else:
                cur = p
    if cur: out.append(cur)
    return [c for c in out if len(c) >= MIN_CHARS] or ([text[:max_c]] if len(text) >= MIN_CHARS else [])

def page_boilerplate(state, threshold=0.25):
    """Lines that recur across many rendered pages are site chrome.

    The announcement ticker ("EAPCET Code: VJEC ... NAAC A++ Accredited ...")
    sits outside <nav>, so render.py cannot strip it structurally - it survives
    on all 105 pages. Counting how many pages a line appears on finds it
    without hard-coding the strings, which would rot at the next redesign.
    """
    counts, pages = Counter(), 0
    for sid, st in state.items():
        snap = st.get("snapshot", "")
        if not snap.endswith(".html"):
            continue
        p = os.path.join(SNAP, snap)
        if not os.path.exists(p):
            continue
        pages += 1
        with open(p, encoding="utf-8") as f:
            counts.update({l.strip() for l in f if l.strip()})
    if pages < 4:
        return set()
    cut = max(2, int(pages * threshold))
    # Only short lines: a long paragraph repeating across pages is real shared
    # content (a policy statement), not chrome.
    return {l for l, n in counts.items() if n >= cut and len(l) < 120}


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def office_pages(path):
    """Text of a .docx or .xlsx, as [(page_no, text)]. Standard library only:
    both are zip files of XML, and the text is plainly in them, so a parsing
    dependency would buy nothing. A .docx is one "page" (it has no fixed
    pages); each worksheet of an .xlsx is one, which is what a citation should
    point at. Table rows are kept on one line, cells separated by " | ", so a
    row's values stay next to each other in a chunk."""
    import zipfile, xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as z:
        if path.endswith(".docx"):
            body = ET.fromstring(z.read("word/document.xml")).find(_W + "body")
            lines = []
            for el in body:
                if el.tag == _W + "p":
                    t = "".join(x.text or "" for x in el.iter(_W + "t")).strip()
                    if t:
                        lines.append(t)
                elif el.tag == _W + "tbl":
                    for tr in el.iter(_W + "tr"):
                        cells = [" ".join("".join(x.text or "" for x in p.iter(_W + "t"))
                                          for p in tc.iter(_W + "p")).strip()
                                 for tc in tr.iter(_W + "tc")]
                        if any(cells):
                            lines.append(" | ".join(cells))
            return [(1, "\n".join(lines))]
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_S + "t")))
        sheets = sorted((n for n in z.namelist()
                         if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        pages = []
        for no, name in enumerate(sheets, 1):
            rows = []
            for row in ET.fromstring(z.read(name)).iter(_S + "row"):
                cells = []
                for c in row.iter(_S + "c"):
                    v = c.find(_S + "v")
                    if c.get("t") == "s" and v is not None:
                        cells.append(shared[int(v.text)])
                    elif c.get("t") == "inlineStr":
                        cells.append("".join(t.text or "" for t in c.iter(_S + "t")))
                    elif v is not None:
                        cells.append(v.text or "")
                cells = [x.strip() for x in cells if x and x.strip()]
                if cells:
                    rows.append(" | ".join(cells))
            pages.append((no, "\n".join(rows)))
        return pages


def extract_one(sid, st, meta, boiler=frozenset()):
    """Chunks for a single document. Returns (chunks, status) where status is
    'ok', 'empty' (scanned/no text - needs OCR, never silently dropped) or
    'fail' (parse error)."""
    path = os.path.join(SNAP, st["snapshot"])
    if not os.path.exists(path):
        return [], "skip"
    m = meta.get(sid, {})
    try:
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            pages = [(n + 1, p.extract_text() or "") for n, p in enumerate(reader.pages)]
        elif path.endswith(".html"):
            # render.py already stripped nav/chrome and stored plain text, so
            # there is no markup left to parse. A page is one "page" for
            # citation purposes - there is nothing finer to point at.
            with open(path, encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f]
            # A page snapshot must be render.py's text. Raw markup means the
            # SPA shell was fetched over plain HTTP instead - identical for
            # every route, and its meta tags once replaced all 106 pages in
            # the knowledge base (INGEST-004). It is not this page's content.
            if lines and lines[0].lstrip().lower().startswith(("<!doctype", "<html")):
                return [], "shell"
            pages = [(1, "\n".join(l for l in lines if l.strip() not in boiler))]
        elif path.endswith((".docx", ".xlsx")):
            pages = office_pages(path)
        else:
            return [], "skip"
    except Exception:
        return [], "fail"
    ocr = False
    if path.endswith(".pdf") and len(normalize("\n\n".join(t for _, t in pages))) < MIN_CHARS:
        # No text layer: a scan. Read the pixels instead of dropping it - the
        # current academic calendars are all scans (see preprocessing/ocr.py).
        global _ocr_spent
        from preprocessing.ocr import ocr_pdf, is_cached
        if _ocr_spent >= OCR_BUDGET_S and not is_cached(path):
            return [], "deferred"
        t0 = time.time()
        read = ocr_pdf(path)
        _ocr_spent += time.time() - t0
        if read:
            pages, ocr = read, True
    if len(normalize("\n\n".join(t for _, t in pages))) < MIN_CHARS:
        return [], "empty"
    out = []
    for pno, ptext in pages:
        ptext = normalize(ptext)
        if len(ptext) < MIN_CHARS:
            continue
        for ci, c in enumerate(chunk(ptext)):
            out.append({
                "chunk_id": hashlib.sha1(f"{sid}:{pno}:{ci}".encode()).hexdigest()[:16],
                "source_id": sid, "url": st["url"],
                "category": m.get("category", "general"),
                "document_type": m.get("document_type", "pdf"),
                "visibility": m.get("visibility", "PUBLIC"),
                "audience": m.get("audience", "student|faculty"),
                "page": pno, "text": c,
                # Which snapshot these chunks were made from. Lets every run
                # prove the knowledge base matches what was last fetched.
                "src_hash": st.get("content_hash"),
                # OCR text is a reading of pixels, not the publisher's own
                # characters. Kept visible so a citation can say so.
                **({"ocr": True} if ocr else {}),
            })
    return out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-ids", help="file of source_ids; re-extract just these "
                                       "and merge into the existing chunk file")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    chunks_path = os.path.join(OUT, "rag_chunks.jsonl")
    meta = {r["source_id"]: r for r in csv.DictReader(open(CATALOG, encoding="utf-8"))}
    state = json.load(open(STATE, encoding="utf-8"))

    targets = {s for s, st in state.items() if st.get("snapshot")}
    kept = []
    # Last-known-good chunks per source. A document that fails to parse this
    # run keeps what it had - in a full run as well as an incremental one.
    # "A failed fetch never deletes last-known-good state" has to hold for
    # extraction too, or a transient parse error silently deletes knowledge.
    processed = json.load(open(EXTRACTED)) if os.path.exists(EXTRACTED) else {}
    previous = {}
    if os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as f:
            for l in f:
                c = json.loads(l)
                previous.setdefault(c["source_id"], []).append(c)
    if a.only_ids:
        # Incremental: re-extract only the named documents and splice their
        # chunks back in. Re-reading 628 PDFs because one changed is the
        # difference between a refresh that takes seconds and one that takes
        # minutes, and the pipeline runs unattended.
        with open(a.only_ids, encoding="utf-8") as f:
            wanted = {l.strip() for l in f if l.strip()}
        # Plus every source whose chunks were NOT made from its current
        # snapshot. Incremental extraction used to assume chunks always match
        # their snapshot; on 2026-09-30 three pages were re-rendered outside a
        # cycle, came back "unchanged" in the next one, and kept chunks made
        # from the old SPA shell (INGEST-005). Now it is checked, not assumed:
        # any mismatch is re-extracted, so the knowledge base heals itself.
        # Chunks from before src_hash existed count as a mismatch once.
        stale = {s for s in targets
                 if state[s].get("status") not in ("retired", "not_found")
                 and processed.get(s) != state[s].get("content_hash")}
        if stale - wanted:
            print(f"  {len(stale - wanted)} source(s) never extracted from their current "
                  f"snapshot - extracting them too")
        backlog = set(json.load(open(OCR_BACKLOG))) if os.path.exists(OCR_BACKLOG) else set()
        targets &= wanted | stale | backlog
        if os.path.exists(chunks_path):
            with open(chunks_path, encoding="utf-8") as f:
                kept = [json.loads(l) for l in f
                        if json.loads(l)["source_id"] not in targets]
        # Drop chunks only for sources fetch.py has explicitly RETIRED (gone
        # from the catalog). Absence from sync_state.json means "not fetched on
        # this machine yet", not "deleted" - treating those as retired would
        # silently wipe the knowledge base on a fresh clone, which is exactly
        # what "a failed fetch must never delete last-known-good knowledge"
        # exists to prevent.
        retired = {sid for sid, st in state.items() if st.get("status") == "retired"}
        kept = [c for c in kept if c["source_id"] not in retired]

    boiler = page_boilerplate(state)
    if boiler:
        print(f"  page chrome: {len(boiler)} recurring lines stripped")

    nchunks = ndocs = nfail = nshell = ncarried = 0
    deferred = []
    empty, fresh = [], []
    items = sorted(targets)
    for i, sid in enumerate(items):
        rows, status = extract_one(sid, state[sid], meta, boiler)
        if status in ("ok", "empty", "shell"):
            processed[sid] = state[sid].get("content_hash")
        if status == "deferred":
            deferred.append(sid)
        if status in ("fail", "skip", "deferred") and previous.get(sid):
            fresh.extend(previous[sid])      # unreadable now; keep what we had
            ncarried += 1
        if status == "fail":
            nfail += 1
        elif status == "shell":
            nshell += 1
        elif status == "empty":
            empty.append(sid)
        elif status == "ok":
            ndocs += 1
            fresh.extend(rows)
            nchunks += len(rows)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} docs, {nchunks} chunks", flush=True)

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in kept + fresh:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total = len(kept) + len(fresh)
    mode = "incremental" if a.only_ids else "full"
    print(f"\nDONE ({mode})  processed={len(items)} docs_with_text={ndocs} "
          f"new_chunks={nchunks} carried_over={len(kept)} total={total} "
          f"parse_failures={nfail} empty_or_scanned={len(empty)} "
          f"unrendered_pages={nshell} kept_last_good={ncarried}")
    if nshell:
        print(f"  !! {nshell} page snapshot(s) are the raw SPA shell, not rendered text - "
              f"excluded. Run ingestion/crawler/render.py (needs playwright).")
    retired = {s for s, st in state.items() if st.get("status") == "retired"}
    with open(EXTRACTED + ".tmp", "w") as f:
        json.dump({s: h for s, h in processed.items() if s not in retired}, f)
    os.replace(EXTRACTED + ".tmp", EXTRACTED)

    # The backlog is exactly what was deferred this run: everything read this
    # time has left it, and anything not targeted keeps its place.
    left = []
    if os.path.exists(OCR_BACKLOG):
        left = [s for s in json.load(open(OCR_BACKLOG)) if s not in targets]
    backlog = sorted(set(left + deferred))
    with open(OCR_BACKLOG, "w") as f:
        json.dump(backlog, f, indent=0)
    print(f"  OCR: {_ocr_spent:.0f}s spent of {OCR_BUDGET_S:.0f}s budget | "
          f"backlog {len(backlog)} scanned document(s) for later runs")
    if empty:
        prev = []
        ep = os.path.join(OUT, "empty_sources.json")
        if a.only_ids and os.path.exists(ep):
            prev = [s for s in json.load(open(ep)) if s not in targets]
        json.dump(sorted(set(prev + empty)), open(ep, "w"), indent=1)
        print(f"  (logged {len(empty)} empty/scanned PDFs - these need OCR, not silently dropped)")

if __name__ == "__main__":
    main()
