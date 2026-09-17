"""Stage 2: fetch + snapshot + change detection.

Re-runnable and incremental. Keeps a raw snapshot and a normalized-content
hash per source. Unchanged sources are skipped. A failed fetch NEVER deletes
last-known-good state. Stdlib only.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, sys, csv, time
import urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(ROOT, "data_catalog", "catalog.csv")
SNAP = os.path.join(ROOT, "ingestion", "snapshots")
STATE = os.path.join(ROOT, "ingestion", "state", "sync_state.json")
UA = "VJLM-research-crawler (VNRVJIET academic project)"
# Politeness comes from configs/ingestion.yaml (request_delay_seconds). It is
# passed in rather than read here so this stays stdlib-only. The effective rate
# is WORKERS/DELAY requests per second - with the pipeline now running
# unattended on every app start, that budget is against the institution's own
# infrastructure, so the default is deliberately conservative.
WORKERS, DELAY = 4, 1.0

def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def norm_hash(data, kind):
    """Fingerprint the CONTENT, not the bytes. For PDFs the raw bytes are the
    content. For HTML we would strip markup first (done at extract stage)."""
    return hashlib.sha256(data).hexdigest()

def fetch_one(row, state):
    url, sid = row["canonical_url"], row["source_id"]
    prev = state.get(sid, {})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if prev.get("etag"):
        req.add_header("If-None-Match", prev["etag"])
    try:
        time.sleep(DELAY)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            etag = r.headers.get("ETag")
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return sid, dict(prev, status="unchanged", last_seen=time.time())
        return sid, dict(prev, status="error", last_error=f"HTTP {e.code}", last_seen=time.time())
    except Exception as e:
        return sid, dict(prev, status="error", last_error=str(e)[:120], last_seen=time.time())

    h = norm_hash(data, row["document_type"])
    if h == prev.get("content_hash"):
        return sid, dict(prev, status="unchanged", last_seen=time.time(), etag=etag or prev.get("etag"))

    path = os.path.join(SNAP, sid + (".pdf" if row["document_type"] == "pdf" else ".html"))
    with open(path, "wb") as f:
        f.write(data)
    return sid, {
        "source_id": sid, "url": url, "snapshot": os.path.basename(path),
        "content_hash": h, "etag": etag, "bytes": len(data),
        "status": "new" if not prev else "changed",
        "last_seen": time.time(), "last_success": time.time(),
        "document_version": prev.get("document_version", 0) + 1, "last_error": None,
    }

def main():
    global DELAY, WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("type", nargs="?", help="only this document_type (pdf|html)")
    ap.add_argument("--only-ids", help="file with one source_id per line; fetch just these")
    ap.add_argument("--delay", type=float, default=DELAY, help="seconds between requests per worker")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    DELAY, WORKERS = a.delay, a.workers

    os.makedirs(SNAP, exist_ok=True); os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(CATALOG, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = all_rows
    if a.type:
        rows = [r for r in rows if r["document_type"] == a.type]
    if a.only_ids:
        with open(a.only_ids, encoding="utf-8") as f:
            wanted = {l.strip() for l in f if l.strip()}
        rows = [r for r in rows if r["source_id"] in wanted]
    state = load_state()
    counts, done = {}, 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_one, r, state): r for r in rows}
        for fut in cf.as_completed(futs):
            sid, st = fut.result()
            state[sid] = st
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}  {counts}", flush=True)
    # Reconciliation is against the FULL catalog, never the filtered subset:
    # a partial run (--only-ids, or a single document_type) must not retire
    # every source it simply did not look at this time.
    ids = {r["source_id"] for r in all_rows}
    for sid, st in state.items():
        if sid not in ids and st.get("status") != "retired":
            st["status"] = "retired"; counts["retired"] = counts.get("retired", 0) + 1
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    total = sum(v.get("bytes", 0) for v in state.values() if isinstance(v, dict))
    print(f"\nDONE {done} sources | {counts} | snapshot corpus {total/1e6:.1f} MB")

if __name__ == "__main__":
    main()
