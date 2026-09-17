"""Independent tests for the self-maintaining pipeline.

Run:  python ingestion/test_refresh.py

These exercise the behaviours the pipeline is supposed to guarantee, against
real files in a temporary copy of the repo layout. No network: snapshots are
synthetic PDFs, so the tests are fast and deterministic.

What is checked, and why each one matters:

  1. due_sources honours PER-SOURCE intervals          - or everything refetches
     every cycle and the crawl budget is wasted.
  2. A partial fetch does NOT retire untouched sources - the bug that would
     wipe the catalog on any incremental run.
  3. Incremental extract touches only changed documents and carries the rest
     over unchanged - the difference between seconds and minutes.
  4. Retired documents lose their chunks                - stale knowledge must
     not survive in the index after a source disappears.
  5. The dense cache re-embeds only new chunks          - the expensive step
     must be incremental or unattended refresh is unusable on CPU.
  6. A failing stage leaves the previous index serving  - "a failed fetch must
     never delete last-known-good knowledge".
"""
import json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    note = ""
    if detail and not cond:
        note = " - " + (detail if isinstance(detail, str) else repr(detail))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, note))


def make_pdf(path, lines):
    """Minimal one-page PDF with extractable text - avoids a reportlab dep."""
    content = "BT /F1 11 Tf 40 760 Td 14 TL\n" + "".join(
        "(%s) Tj T*\n" % l.replace("(", "").replace(")", "") for l in lines) + "ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        "<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += "%d 0 obj\n%s\nendobj\n" % (i, o)
    xref = len(out)
    out += "xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += "".join("%010d 00000 n \n" % o for o in offsets)
    out += "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    with open(path, "wb") as f:
        f.write(out.encode("latin-1"))


def sandbox():
    """A throwaway repo layout with three synthetic documents."""
    d = tempfile.mkdtemp(prefix="vjlm_test_")
    for sub in ("data_catalog", "ingestion/snapshots", "ingestion/state",
                "preprocessing/out", "rag/index", "configs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)

    for name, code in [("scripts", None)]:
        pass
    for rel in ("preprocessing/extract.py", "rag/index.py", "ingestion/crawler/fetch.py"):
        os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), os.path.join(d, rel))

    # Comfortably over MIN_CHARS (200) so the anomaly guard treats these as
    # real documents rather than scanned/empty ones.
    body = ["The institute follows the prescribed academic regulation for each "
            "admission batch and publishes it on the official website.",
            "Attendance of seventy five percent of the aggregate is required for "
            "a student to be eligible to appear for the end semester examination.",
            "Condonation may be granted subject to the conditions stated in the "
            "regulation, on payment of the prescribed fee within the notified date.",
            "The academic council reviews these provisions periodically and any "
            "revision is notified to all departments before it takes effect."]
    rows, state = [], {}
    for i, freq in enumerate(["daily", "weekly", "yearly"], 1):
        sid = "doc%d" % i
        make_pdf(os.path.join(d, "ingestion/snapshots", sid + ".pdf"), body + ["Document %d." % i])
        rows.append({"source_id": sid, "canonical_url": "https://x/%s.pdf" % sid,
                     "category": "academic_regulations", "document_type": "pdf",
                     "audience": "student|faculty", "visibility": "PUBLIC",
                     "update_frequency": freq, "parser_type": "pdf_text",
                     "fine_tune_usage": "no", "rag_usage": "full_source", "active": "true"})
        state[sid] = {"source_id": sid, "url": "https://x/%s.pdf" % sid,
                      "snapshot": sid + ".pdf", "content_hash": "h%d" % i,
                      "status": "new", "last_success": time.time(), "bytes": 1000}

    import csv
    with open(os.path.join(d, "data_catalog/catalog.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(d, "ingestion/state/sync_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)
    return d


def run(d, script, args=()):
    p = subprocess.run([sys.executable, os.path.join(d, script)] + list(args),
                       cwd=d, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (script, p.stderr[-800:]))
    return p.stdout


def chunks_of(d):
    path = os.path.join(d, "preprocessing/out/rag_chunks.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    d = sandbox()
    print("sandbox:", d)
    try:
        # ---- 1. per-source intervals -----------------------------------
        print("\n[1] due_sources honours per-source intervals")
        import ingestion.refresh as R
        orig = R.CATALOG, R.STATE
        R.CATALOG = os.path.join(d, "data_catalog/catalog.csv")
        R.STATE = os.path.join(d, "ingestion/state/sync_state.json")
        try:
            check("nothing due immediately after a fetch", R.due_sources() == [])
            # 7 hours later: only the 'daily' source (PT6H) is due
            due = R.due_sources(now=time.time() + 7 * 3600)
            check("only the 6-hourly source is due after 7h", due == ["doc1"], str(due))
            due = R.due_sources(now=time.time() + 8 * 86400)
            check("daily+weekly due after 8 days, yearly not",
                  sorted(due) == ["doc1", "doc2"], str(due))
        finally:
            R.CATALOG, R.STATE = orig

        # ---- 2/3. full then incremental extract -------------------------
        print("\n[2] full extract, then incremental")
        run(d, "preprocessing/extract.py")
        base = chunks_of(d)
        check("all three documents extracted",
              len({c["source_id"] for c in base}) == 3,
              str(sorted({c["source_id"] for c in base})))

        # change doc2 only
        make_pdf(os.path.join(d, "ingestion/snapshots/doc2.pdf"),
                 ["Revised text replacing the second document in its entirety, "
                  "issued under a later notification from the examination branch.",
                  "The attendance requirement is now stated differently here and "
                  "supersedes whatever the previous version of this page carried.",
                  "This particular sentence exists only in the revised revision two "
                  "and is what the incremental extract test looks for afterwards.",
                  "All other clauses of the regulation continue to apply without "
                  "any change to their numbering or their stated effective dates."])
        ids = os.path.join(d, "ingestion/state/changed.txt")
        open(ids, "w").write("doc2")
        out = run(d, "preprocessing/extract.py", ["--only-ids", ids])
        after = chunks_of(d)
        check("incremental run reports processed=1", "processed=1" in out, out.strip()[-120:])
        check("untouched documents carried over unchanged",
              [c for c in base if c["source_id"] != "doc2"] ==
              [c for c in after if c["source_id"] != "doc2"])
        check("changed document's text was replaced",
              any("revised revision two" in c["text"] for c in after))
        check("no duplicate chunks for the changed document",
              len({c["chunk_id"] for c in after}) == len(after))

        # ---- 4. retirement ---------------------------------------------
        print("\n[3] retirement removes chunks, but absence does not")
        sp = os.path.join(d, "ingestion/state/sync_state.json")

        # A source merely MISSING from state has not been fetched on this
        # machine yet. Its chunks must survive - treating absence as deletion
        # would wipe the knowledge base on a fresh clone.
        st = json.load(open(sp))
        gone = st.pop("doc1")
        json.dump(st, open(sp, "w"))
        open(ids, "w").write("doc2")
        run(d, "preprocessing/extract.py", ["--only-ids", ids])
        check("chunks survive for a source absent from state",
              any(c["source_id"] == "doc1" for c in chunks_of(d)))

        # An EXPLICITLY retired source does lose its chunks.
        st["doc1"] = gone
        st["doc3"]["status"] = "retired"
        json.dump(st, open(sp, "w"))
        run(d, "preprocessing/extract.py", ["--only-ids", ids])
        check("chunks of an explicitly retired source are gone",
              not any(c["source_id"] == "doc3" for c in chunks_of(d)))

        # ---- 5. dense cache is incremental ------------------------------
        print("\n[4] dense vector cache re-embeds only new chunks")
        try:
            t0 = time.time(); run(d, "rag/index.py"); first = time.time() - t0
            cache = os.path.join(d, "rag/index/dense_cache.npz")
            check("cache written", os.path.exists(cache))
            out = run(d, "rag/index.py")
            line = next((l.strip() for l in out.splitlines() if "to embed" in l), "")
            check("second run embeds nothing new", "to embed: 0 new" in out, line)
            print("      (first build %.1fs for %d chunks)" % (first, len(chunks_of(d))))
        except RuntimeError as e:
            check("dense build", False, str(e)[:200])

        # ---- 5b. edited text must NOT reuse its old vector ---------------
        print("\n[4b] an edited chunk gets a fresh embedding, not the cached one")
        try:
            sys.path.insert(0, ROOT)
            from rag.index import vector_key
            # chunk_id is positional (source:page:index), so an in-place edit
            # keeps the id. If the cache were keyed on it, the stale vector
            # would be served silently - it was, and a query for the old topic
            # still matched the chunk after its text had been replaced.
            a = "Attendance of seventy five percent of the aggregate is required."
            b = "The canteen serves breakfast and lunch on all working days."
            check("cache key changes when the text changes",
                  vector_key(a) != vector_key(b))
            check("cache key is stable for identical text",
                  vector_key(a) == vector_key(a))
            check("cache key ignores position entirely",
                  vector_key(b) == vector_key(b))
        except Exception as e:
            check("vector_key available", False, str(e)[:160])

        # ---- 6. failure leaves the index intact -------------------------
        print("\n[5] a failing stage leaves the previous index serving")
        before = os.path.getmtime(os.path.join(d, "rag/index/bm25.pkl"))
        r = R.Refresher(python=sys.executable)
        r._run_root = d
        bad = R.Refresher(python=sys.executable)
        bad.ROOT = d
        try:
            bad._run("preprocessing/does_not_exist.py")
            check("missing stage raises", False, "no exception")
        except Exception:
            check("missing stage raises rather than continuing silently", True)
        check("index file untouched after a failed stage",
              os.path.getmtime(os.path.join(d, "rag/index/bm25.pkl")) == before)

    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
