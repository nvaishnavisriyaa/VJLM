"""Stage 3: extract text from snapshots -> normalized, chunked JSONL for RAG.

Structure-aware chunking with page provenance so every chunk can cite the
exact document and page it came from.
"""
import json, os, re, sys, csv, hashlib
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

def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {r["source_id"]: r for r in csv.DictReader(open(CATALOG, encoding="utf-8"))}
    state = json.load(open(STATE, encoding="utf-8"))
    fout = open(os.path.join(OUT, "rag_chunks.jsonl"), "w", encoding="utf-8")
    nchunks = ndocs = nfail = 0
    empty = []
    items = [(s, st) for s, st in state.items() if st.get("snapshot")]
    for i, (sid, st) in enumerate(items):
        path = os.path.join(SNAP, st["snapshot"])
        if not os.path.exists(path): continue
        m = meta.get(sid, {})
        try:
            if path.endswith(".pdf"):
                reader = PdfReader(path)
                pages = [(n + 1, p.extract_text() or "") for n, p in enumerate(reader.pages)]
            else:
                continue
        except Exception as e:
            nfail += 1; continue
        full = normalize("\n\n".join(t for _, t in pages))
        if len(full) < MIN_CHARS:
            empty.append(sid); continue          # anomaly guard: scanned/empty PDF
        ndocs += 1
        for pno, ptext in pages:
            ptext = normalize(ptext)
            if len(ptext) < MIN_CHARS: continue
            for ci, c in enumerate(chunk(ptext)):
                cid = hashlib.sha1(f"{sid}:{pno}:{ci}".encode()).hexdigest()[:16]
                fout.write(json.dumps({
                    "chunk_id": cid, "source_id": sid, "url": st["url"],
                    "category": m.get("category", "general"),
                    "document_type": m.get("document_type", "pdf"),
                    "visibility": m.get("visibility", "PUBLIC"),
                    "audience": m.get("audience", "student|faculty"),
                    "page": pno, "text": c,
                }, ensure_ascii=False) + "\n")
                nchunks += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} docs, {nchunks} chunks", flush=True)
    fout.close()
    print(f"\nDONE  docs_with_text={ndocs}  chunks={nchunks}  parse_failures={nfail}  "
          f"empty_or_scanned={len(empty)}")
    if empty:
        json.dump(empty, open(os.path.join(OUT, "empty_sources.json"), "w"), indent=1)
        print(f"  (logged {len(empty)} empty/scanned PDFs - these need OCR, not silently dropped)")

if __name__ == "__main__":
    main()
