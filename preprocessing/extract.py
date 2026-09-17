"""Stage 3: extract text from snapshots -> normalized, chunked JSONL for RAG.

Structure-aware chunking with page provenance so every chunk can cite the
exact document and page it came from.
"""
import argparse, json, os, re, sys, csv, hashlib
from collections import Counter
from pypdf import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "ingestion", "snapshots")
STATE = os.path.join(ROOT, "ingestion", "state", "sync_state.json")
CATALOG = os.path.join(ROOT, "data_catalog", "catalog.csv")
OUT = os.path.join(ROOT, "preprocessing", "out")
MIN_CHARS, MAX_CHARS, OVERLAP = 200, 1100, 150

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
            pages = [(1, "\n".join(l for l in lines if l.strip() not in boiler))]
        else:
            return [], "skip"
    except Exception:
        return [], "fail"
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
    if a.only_ids:
        # Incremental: re-extract only the named documents and splice their
        # chunks back in. Re-reading 628 PDFs because one changed is the
        # difference between a refresh that takes seconds and one that takes
        # minutes, and the pipeline runs unattended.
        with open(a.only_ids, encoding="utf-8") as f:
            wanted = {l.strip() for l in f if l.strip()}
        targets &= wanted
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

    nchunks = ndocs = nfail = 0
    empty, fresh = [], []
    items = sorted(targets)
    for i, sid in enumerate(items):
        rows, status = extract_one(sid, state[sid], meta, boiler)
        if status == "fail":
            nfail += 1
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
          f"parse_failures={nfail} empty_or_scanned={len(empty)}")
    if empty:
        prev = []
        ep = os.path.join(OUT, "empty_sources.json")
        if a.only_ids and os.path.exists(ep):
            prev = [s for s in json.load(open(ep)) if s not in targets]
        json.dump(sorted(set(prev + empty)), open(ep, "w"), indent=1)
        print(f"  (logged {len(empty)} empty/scanned PDFs - these need OCR, not silently dropped)")

if __name__ == "__main__":
    main()
