"""Stage 2b: render the HTML pages.

The VNRVJIET site is a client-rendered SPA - every route returns the same ~4 KB
shell over plain HTTP, so `fetch.py` captures nothing usable for the 106 page
routes (docs/phase1_site_recon.md, finding 1). A headless browser is therefore
the PRIMARY fetch path for pages, not a fallback, whatever
configs/ingestion.yaml's `headless_fallback` name suggests.

A local headless browser is not an external API, so this stays inside the
INGEST-001 constraint: nothing is sent anywhere, the page is just executed
locally the way a visitor's browser would.

Change detection hashes the EXTRACTED TEXT, not the rendered HTML. The config
is explicit about why: "Hashing raw HTML would make a CSS redesign look like a
knowledge change and force pointless re-embedding." A React build changes class
names and bundle hashes on every deploy, so raw-HTML hashing would mark all 106
pages changed every single time.

Re-runnable and incremental, same contract as fetch.py: unchanged pages are
skipped, and a failed render never deletes last-known-good state.
"""
import argparse, csv, hashlib, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(ROOT, "data_catalog", "catalog.csv")
SNAP = os.path.join(ROOT, "ingestion", "snapshots")
STATE = os.path.join(ROOT, "ingestion", "state", "sync_state.json")
UA = "VJLM-research-crawler (VNRVJIET academic project)"

# Chrome that a real visitor would not distinguish from a normal tab, but
# identified honestly in the UA so the site owner can see who we are.
NAV_TIMEOUT_MS = 45000
MIN_CHARS = 200           # below this, treat the render as a failure

# Hydration: "networkidle" fires when the network quiets, which on a React app
# can be BEFORE the components have painted their data. A fixed sleep is a
# guess that is either wasteful or wrong. Waiting for a named element would be
# accurate but needs a selector per page, and there are 106 heterogeneous ones.
# Instead poll the rendered text length until it stops growing - generic, and
# it exits as soon as the page is actually done.
STABLE_POLL_MS = 250
STABLE_REQUIRED = 3       # consecutive equal samples before we call it settled
STABLE_TIMEOUT_MS = 12000

# A partial hydration is more dangerous than an empty one: it clears MIN_CHARS
# and silently REPLACES good content with less of it. configs/ingestion.yaml
# already specifies max_content_shrink_ratio for exactly this and nothing ever
# implemented it.
MAX_SHRINK_RATIO = 0.5

# sync.retry_transient_failures in configs/ingestion.yaml. RETRIES counts the
# first attempt, so 3 means the original plus two retries.
RETRIES = 3
RETRY_BACKOFF_S = 2.0


def wait_for_stable_text(page):
    """Block until the page's text stops changing, or the timeout expires.
    Returns True if it settled, False if it was still moving."""
    last, stable, waited = -1, 0, 0
    while waited < STABLE_TIMEOUT_MS:
        n = page.evaluate("document.body ? document.body.innerText.length : 0")
        if n == last and n > 0:
            stable += 1
            if stable >= STABLE_REQUIRED:
                return True
        else:
            stable = 0
        last = n
        page.wait_for_timeout(STABLE_POLL_MS)
        waited += STABLE_POLL_MS
    return False


def extract_text(html):
    """Main content, with navigation and chrome removed.

    Structural, not selector-brittle (MASTER_PLAN 6.4): try semantic containers
    in order, and always strip the elements that repeat on every page - those
    are what create duplicate chunks across the whole site."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "svg", "form", "iframe", "button"]):
        tag.decompose()
    for tag in soup.select("[role=navigation], .sr-only, .visually-hidden"):
        tag.decompose()
    # NOT [aria-hidden=true]: it marks decorative icons, but this site also
    # puts it on the homepage's scroll-animation wrapper, so stripping it
    # wholesale deleted the entire page. Decorative nodes carry no text anyway.
    # Pick the RICHEST semantic container, not the first one that exists.
    # On /about/anti-ragging the real content sits outside <main>, which holds
    # only a 190-char heading - taking <main> first silently lost the page.
    candidates = [c for c in (soup.find("main"), soup.find("article"), soup.body) if c]
    if not candidates:
        return "", ""
    node = max(candidates, key=lambda c: len(c.get_text(" ", strip=True)))
    title = (soup.title.get_text(strip=True) if soup.title else "")
    text = node.get_text("\n", strip=True)
    lines, seen, out = text.splitlines(), set(), []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # a nav label repeated verbatim within one page is chrome, not content
        if len(ln) < 40 and ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return title, "\n".join(out)


def render_all(rows, state, delay=1.0, headful=False):
    from playwright.sync_api import sync_playwright

    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1366, "height": 900})
        page = ctx.new_page()
        # Only fonts and media are blocked. Blocking IMAGES broke the homepage:
        # its carousel script throws on a failed image load and never renders
        # the content, so the page came back empty. Saving a few bytes is not
        # worth silently losing a page.
        page.route("**/*", lambda r: r.abort()
                   if r.request.resource_type in ("font", "media") else r.continue_())

        for i, row in enumerate(rows, 1):
            sid, url = row["source_id"], row["canonical_url"]
            prev = state.get(sid, {})
            # configs/ingestion.yaml has specified retry_transient_failures: 3
            # since Phase 3 and nothing implemented it. A navigation timeout is
            # usually transient - in one 106-page run 6 pages timed out and all
            # 6 succeeded on a later cycle - so retrying in-run turns a week of
            # staleness into a few extra seconds.
            html = title = text = None
            err = None
            for attempt in range(RETRIES):
                try:
                    page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
                    settled = wait_for_stable_text(page)
                    html = page.content()
                    title, text = extract_text(html)
                    if not settled:
                        print("  [%d/%d] WARN  %s still changing at timeout"
                              % (i, len(rows), url), flush=True)
                    err = None
                    break
                except Exception as e:
                    err = e
                    if attempt + 1 < RETRIES:
                        print("  [%d/%d] retry %d/%d %s (%s)"
                              % (i, len(rows), attempt + 1, RETRIES - 1, url,
                                 str(e).splitlines()[0][:50]), flush=True)
                        time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            if err is not None:
                # Fail open: keep whatever we already had for this page.
                results[sid] = dict(prev, status="error", last_error=str(err)[:120],
                                    last_seen=time.time())
                print("  [%d/%d] ERROR %s (%s)" % (i, len(rows), url,
                                                   str(err).splitlines()[0][:60]), flush=True)
                continue

            if len(text) < MIN_CHARS:
                results[sid] = dict(prev, status="error", last_seen=time.time(),
                                    last_error="rendered under %d chars" % MIN_CHARS)
                print("  [%d/%d] EMPTY %s (%d chars)" % (i, len(rows), url, len(text)), flush=True)
                continue

            # Anomaly guard: a page that suddenly renders far less than last
            # time is a partial hydration, not an edit. Keep the last known
            # good snapshot rather than overwriting knowledge with a fragment.
            was = prev.get("bytes", 0)
            if was and len(text.encode("utf-8")) < was * MAX_SHRINK_RATIO:
                results[sid] = dict(prev, status="error", last_seen=time.time(),
                                    last_error="shrank %d -> %d bytes; kept previous"
                                               % (was, len(text.encode("utf-8"))))
                print("  [%d/%d] SHRANK %s (%d -> %d bytes, rejected)"
                      % (i, len(rows), url, was, len(text.encode("utf-8"))), flush=True)
                continue

            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h == prev.get("content_hash"):
                results[sid] = dict(prev, status="unchanged", last_seen=time.time(),
                                    last_success=time.time())
                print("  [%d/%d] same  %s" % (i, len(rows), url), flush=True)
            else:
                path = os.path.join(SNAP, sid + ".html")
                # Store the extracted text, not the SPA markup: the markup is
                # regenerated bundle noise, the text is the knowledge.
                with open(path, "w", encoding="utf-8") as f:
                    f.write((title + "\n\n" if title else "") + text)
                results[sid] = {
                    "source_id": sid, "url": url, "snapshot": os.path.basename(path),
                    "content_hash": h, "etag": None, "bytes": len(text.encode("utf-8")),
                    "status": "new" if not prev else "changed",
                    "last_seen": time.time(), "last_success": time.time(),
                    "document_version": prev.get("document_version", 0) + 1,
                    "last_error": None, "title": title,
                }
                print("  [%d/%d] %-7s %s (%d chars)"
                      % (i, len(rows), results[sid]["status"], url, len(text)), flush=True)
            time.sleep(delay)

        ctx.close()
        browser.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-ids", help="file of source_ids to render")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="render at most N pages")
    ap.add_argument("--headful", action="store_true", help="show the browser (debugging)")
    a = ap.parse_args()

    os.makedirs(SNAP, exist_ok=True)
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(CATALOG, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["document_type"] == "html"]
    if a.only_ids:
        with open(a.only_ids, encoding="utf-8") as f:
            wanted = {l.strip() for l in f if l.strip()}
        rows = [r for r in rows if r["source_id"] in wanted]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print("no HTML sources to render")
        return

    state = {}
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            state = json.load(f)

    print("rendering %d page(s) with headless Chromium ..." % len(rows), flush=True)
    t0 = time.time()
    results = render_all(rows, state, a.delay, a.headful)
    state.update(results)

    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE)

    from collections import Counter
    c = Counter(v["status"] for v in results.values())
    print("\nDONE %d pages in %.1fs | %s" % (len(rows), time.time() - t0, dict(c)))


if __name__ == "__main__":
    main()
