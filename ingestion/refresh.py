"""Self-maintaining refresh — the automatic half of Phase 3.

The pipeline stages are re-runnable scripts. This decides WHEN each one needs
to run and drives them, so nobody has to remember to.

It is started by app/server.py, so opening VJLM is what keeps the knowledge
base current. No Windows Task Scheduler entry, no cron, no human step.

Two rules from configs/ingestion.yaml shape this:

  * Interval is PER SOURCE, from that source's update_frequency. An exam
    notification is not rechecked on the same cadence as the college history
    page, so a cycle usually fetches a handful of sources, not 652.

  * Index sync happens in the SAME run as the fetch. "A refresh that updates
    snapshots but leaves the index stale is a silent failure." So extract ->
    enrich -> index run whenever content actually changed, and the serving
    retriever is swapped only once the new index is complete.

Failures never delete last-known-good knowledge: a stage that fails aborts the
cycle and leaves the previous index serving.
"""
import json, os, re, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(ROOT, "data_catalog", "catalog.csv")
STATE = os.path.join(ROOT, "ingestion", "state", "sync_state.json")
REFRESH_STATE = os.path.join(ROOT, "ingestion", "state", "refresh_state.json")
CONFIG = os.path.join(ROOT, "configs", "ingestion.yaml")
DUE_FILE = os.path.join(ROOT, "ingestion", "state", "due_ids.txt")

# Fallbacks if configs/ingestion.yaml cannot be parsed. Same values as the file.
DEFAULT_INTERVALS = {"daily": "PT6H", "weekly": "P1D", "monthly": "P7D",
                     "yearly": "P30D", "unknown": "P7D"}
DISCOVER_INTERVAL_H = 24        # how often to look for NEW documents
CYCLE_SLEEP_S = 900             # 15 min between checks while the app is up

# A deploy makes every page worth re-checking, but the bundle hash changes on
# ANY code change - a CSS tweak or a dependency bump, not just page copy.
# Observed 2026-09-17: the site shipped twice within 15 minutes and all 106
# pages came back byte-identical both times, i.e. ~22 minutes of browser work
# for nothing. This floor stops an active deploy day turning the gate from an
# optimisation into a cost. Per-source intervals still apply underneath.
DEPLOY_RECHECK_MIN_H = 6

ISO = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$")


def parse_duration(s):
    """ISO-8601 duration -> seconds. Only the forms this config uses."""
    m = ISO.match(str(s).strip())
    if not m:
        return 7 * 86400
    d, h, mi = (int(x) if x else 0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60


def load_intervals():
    try:
        import yaml
        with open(CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        raw = cfg["schedule"]["intervals_by_update_frequency"]
        enabled = cfg["schedule"].get("enabled", True)
    except Exception:
        raw, enabled = DEFAULT_INTERVALS, True
    return {k: parse_duration(v) for k, v in raw.items()}, enabled


def crawl_delay():
    """scope.request_delay_seconds - the politeness budget. Honoured here
    because this pipeline now runs unattended."""
    try:
        import yaml
        with open(CONFIG, encoding="utf-8") as f:
            return float(yaml.safe_load(f)["scope"]["request_delay_seconds"])
    except Exception:
        return 1.0


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def all_source_ids():
    """Every active source in the catalog."""
    import csv
    try:
        with open(CATALOG, encoding="utf-8") as f:
            return [r["source_id"] for r in csv.DictReader(f)
                    if r.get("active", "true").lower() == "true"]
    except FileNotFoundError:
        return []


def split_by_type(ids):
    """(pdf_ids, html_ids) - the two fetch paths are different tools."""
    import csv
    kind = {}
    try:
        with open(CATALOG, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                kind[r["source_id"]] = r["document_type"]
    except FileNotFoundError:
        return list(ids), []
    pdfs = [i for i in ids if kind.get(i) == "pdf"]
    pages = [i for i in ids if kind.get(i) == "html"]
    return pdfs, pages


def due_sources(now=None):
    """Which sources are past their own recheck interval. A source that has
    never been fetched is always due."""
    import csv
    now = now or time.time()
    intervals, _ = load_intervals()
    state = read_json(STATE, {})
    due = []
    try:
        with open(CATALOG, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return []
    for r in rows:
        if r.get("active", "true").lower() != "true":
            continue
        st = state.get(r["source_id"]) or {}
        last = st.get("last_success") or st.get("last_seen") or 0
        gap = intervals.get(r.get("update_frequency", "unknown"),
                            intervals.get("unknown", 7 * 86400))
        if now - last >= gap:
            due.append(r["source_id"])
    return due


class Refresher:
    """Owns the refresh cycle and reports what it is doing to the UI."""

    def __init__(self, on_index_rebuilt=None, python=None):
        self.on_index_rebuilt = on_index_rebuilt
        self.python = python or sys.executable
        self._lock = threading.Lock()
        self._thread = None
        self._embedding = None
        self._bundle = None
        self._why = ""
        self._stop = threading.Event()
        self.state = {"phase": "idle", "detail": "", "started": None,
                      "finished": None, "changed": 0, "error": None,
                      "embedding": "idle"}

    # -- reporting -------------------------------------------------------
    def status(self):
        rs = read_json(REFRESH_STATE, {})
        out = dict(self.state)
        out["last_cycle"] = rs.get("last_cycle")
        out["last_discover"] = rs.get("last_discover")
        out["due_now"] = len(due_sources())
        out["running"] = self._lock.locked()
        return out

    def _set(self, phase, detail=""):
        self.state["phase"] = phase
        self.state["detail"] = detail
        print("[refresh] %s %s" % (phase, detail), flush=True)

    # -- stage runner ----------------------------------------------------
    def _run(self, script, args=()):
        """Stream the stage's output instead of swallowing it.

        This runs unattended for tens of minutes; capturing output means a
        stalled crawl looks identical to a working one. Lines are echoed with
        the stage name and the tail is kept for the error message.
        """
        cmd = [self.python, os.path.join(ROOT, script)] + list(args)
        tag = os.path.basename(script)
        tail, lines = [], []
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            tail.append(line)
            del tail[:-6]
            print("    [%s] %s" % (tag, line), flush=True)
            self.state["detail"] = line[:120]
        p.wait()
        if p.returncode != 0:
            raise RuntimeError("%s failed: %s" % (script, " | ".join(tail[-3:])))
        return "\n".join(lines)

    # -- dense vectors ---------------------------------------------------
    def _embed_async(self):
        """Dense embedding runs outside the cycle lock. It is incremental, so
        after the first build this is usually a few seconds - but the first
        build is long on CPU, and blocking every content refresh behind it
        would defeat the point of an automatic pipeline."""
        if self._embedding and self._embedding.is_alive():
            return

        def work():
            try:
                self.state["embedding"] = "running"
                self._run("rag/index.py")      # reuses the per-chunk cache
                if self.on_index_rebuilt:
                    self.on_index_rebuilt()    # pick up dense.npy
                self.state["embedding"] = "done"
            except Exception as e:
                self.state["embedding"] = "failed: %s" % str(e)[:120]

        self._embedding = threading.Thread(target=work, daemon=True, name="vjlm-embed")
        self._embedding.start()

    # -- first run -------------------------------------------------------
    def ensure_index(self):
        """A fresh clone has committed chunks but no index (rag/index/ is
        gitignored). Building it takes seconds and needs no network, so do it
        before any crawling - the app becomes usable immediately instead of
        waiting on a full first sync."""
        index = os.path.join(ROOT, "rag", "index", "bm25.pkl")
        chunks = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")
        if os.path.exists(index) or not os.path.exists(chunks):
            return False
        with self._lock:
            try:
                self._set("indexing", "first run: building index from stored chunks")
                self._run("rag/index.py", ["--bm25-only"])
                if self.on_index_rebuilt:
                    self.on_index_rebuilt()
                self._set("idle", "index ready")
            except Exception as e:
                self.state["error"] = str(e)[:300]
                self._set("error", str(e)[:200])
                return False
        self._embed_async()          # dense follows, outside the lock
        return True

    # -- the cycle -------------------------------------------------------
    def cycle(self, force=False):
        """One pass. Returns True if the index was rebuilt."""
        if not self._lock.acquire(blocking=False):
            return False                      # a cycle is already running
        try:
            self.state.update(started=time.time(), finished=None, error=None, changed=0)
            rs = read_json(REFRESH_STATE, {})
            now = time.time()

            # 1. Has the site shipped? One 4 KB request, and it drives both
            #    discovery and the page recheck below.
            sys.path.insert(0, os.path.join(ROOT, "ingestion", "crawler"))
            from site_version import pages_may_have_changed
            deployed, self._bundle, self._why = pages_may_have_changed()

            # 2. Discovery finds documents that did not exist before, and this
            #    is what actually governs freshness: a new circular gets a new
            #    URL, so it arrives as a NEW source (always due) rather than
            #    waiting out any recheck interval.
            #
            #    Every document URL is referenced from the JS bundle, so an
            #    unchanged bundle means no new documents - discovery can be
            #    skipped. That makes it effectively continuous instead of
            #    daily, at the cost of one 4 KB request, while the 24h timer
            #    stays as a backstop in case the signal is ever wrong.
            stale = now - rs.get("last_discover", 0) >= DISCOVER_INTERVAL_H * 3600
            if force or deployed or stale:
                why = "deploy" if deployed else ("%dh since last" % DISCOVER_INTERVAL_H)
                self._set("discovering", "looking for new documents (%s)" % why)
                self._run("ingestion/crawler/discover.py")
                rs["last_discover"] = now
                write_json(REFRESH_STATE, rs)

            # 3. fetch only what is past its own interval
            due = due_sources(now)

            # A new deploy is a better freshness signal for pages than any
            # fixed interval: the bundle hash changes iff the app changed, so
            # a site update should be picked up now rather than waiting out a
            # 7- or 30-day recheck. Cheap because it is one 4 KB request, and
            # the per-page text hashes still decide whether anything downstream
            # runs - a CSS-only deploy costs renders and no re-indexing.
            since = now - rs.get("last_deploy_recheck", 0)
            if deployed and since < DEPLOY_RECHECK_MIN_H * 3600 and not force:
                # Deploy noted, but we rechecked recently. Deliberately do NOT
                # record the new bundle hash - the next eligible cycle must
                # still see it as changed, or this deploy is lost entirely.
                self._set("fetching", "deploy seen (%s) but rechecked %.1fh ago; "
                                      "pages stay on their own intervals"
                          % (self._why, since / 3600))
                deployed, self._bundle = False, None
            elif deployed:
                _, all_pages = split_by_type(all_source_ids())
                added = [p for p in all_pages if p not in due]
                if added:
                    self._set("fetching", "site deploy detected (%s); %d page%s to recheck"
                              % (self._why, len(added), "" if len(added) == 1 else "s"))
                    due = due + added
                rs["last_deploy_recheck"] = now
                write_json(REFRESH_STATE, rs)

            if not due and not force:
                self._set("idle", "nothing due")
                rs["last_cycle"] = now
                write_json(REFRESH_STATE, rs)
                return False

            self._set("fetching", "%d source%s due" % (len(due), "" if len(due) == 1 else "s"))
            os.makedirs(os.path.dirname(DUE_FILE), exist_ok=True)
            with open(DUE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(due))
            before = read_json(STATE, {})

            # PDFs over plain HTTP; page routes through a headless browser,
            # because this site renders its pages in the client and plain HTTP
            # returns an empty shell for every one of them.
            pdfs, pages = split_by_type(due)
            if pdfs:
                self._run("ingestion/crawler/fetch.py",
                          ["--only-ids", DUE_FILE, "--delay", str(crawl_delay())])
            if pages:
                # The bundle check already ran above. If nothing shipped, the
                # pages that came due on a timer cannot have changed, so skip
                # the browser entirely.
                from site_version import write as save_bundle
                if not deployed and not force:
                    self._set("fetching", "pages skipped - %s" % self._why)
                    pages = []
                else:
                    self._set("rendering", "%d page%s in headless browser (%s)"
                              % (len(pages), "" if len(pages) == 1 else "s", self._why))
                    self._run("ingestion/crawler/render.py",
                              ["--only-ids", DUE_FILE, "--delay", str(crawl_delay())])
                    if self._bundle:
                        save_bundle(self._bundle)
            after = read_json(STATE, {})

            changed = [s for s in due
                       if after.get(s, {}).get("content_hash") != before.get(s, {}).get("content_hash")]
            self.state["changed"] = len(changed)

            # 3. only rebuild when content actually moved - re-extracting 652
            #    PDFs because a timestamp ticked would be wasteful and pointless
            if not changed and not force:
                self._set("idle", "checked %d, nothing changed" % len(due))
                rs["last_cycle"] = now
                write_json(REFRESH_STATE, rs)
                return False

            self._set("extracting", "%d document%s changed" % (len(changed), "" if len(changed) == 1 else "s"))
            with open(DUE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(changed))
            self._run("preprocessing/extract.py", ["--only-ids", DUE_FILE])
            self._set("enriching", "metadata, PII flags, duplicates")
            self._run("preprocessing/enrich.py")

            # BM25 first: it is quick and it is what serves. Dense vectors are
            # incremental (cached per chunk_id) but still the slow part on CPU,
            # so they are built after the retriever is already live rather than
            # holding the update behind them.
            self._set("indexing", "rebuilding the search index")
            self._run("rag/index.py", ["--bm25-only"])
            if self.on_index_rebuilt:
                self._set("reloading", "swapping in the new index")
                self.on_index_rebuilt()
            self._embed_async()

            rs["last_cycle"] = now
            write_json(REFRESH_STATE, rs)
            self._set("idle", "updated %d document%s" % (len(changed), "" if len(changed) == 1 else "s"))
            return True

        except Exception as e:
            # Fail open: the previous snapshots and index keep serving.
            self.state["error"] = str(e)[:300]
            self._set("error", str(e)[:200])
            return False
        finally:
            self.state["finished"] = time.time()
            self._lock.release()

    # -- background loop -------------------------------------------------
    def _loop(self, initial_delay):
        time.sleep(initial_delay)
        self.ensure_index()          # make the app usable before crawling
        while not self._stop.is_set():
            self.cycle()
            self._stop.wait(CYCLE_SLEEP_S)

    def start(self, initial_delay=2.0):
        """Called by the app at startup. Non-blocking: the UI serves from the
        existing index immediately and the refresh catches up behind it."""
        intervals, enabled = load_intervals()
        if not enabled:
            self._set("disabled", "schedule.enabled is false in configs/ingestion.yaml")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, args=(initial_delay,),
                                        daemon=True, name="vjlm-refresh")
        self._thread.start()

    def trigger(self, force=False):
        """Fire a cycle now, in the background, without blocking the request."""
        if self._lock.locked():
            return False
        threading.Thread(target=self.cycle, kwargs={"force": force},
                         daemon=True, name="vjlm-refresh-once").start()
        return True

    def stop(self):
        self._stop.set()


if __name__ == "__main__":
    force = "--force" in sys.argv
    r = Refresher()
    if "--status" in sys.argv:
        print(json.dumps(r.status(), indent=2, default=str))
    else:
        print("running one refresh cycle (force=%s)" % force)
        r.cycle(force=force)
        print(json.dumps(r.status(), indent=2, default=str))
