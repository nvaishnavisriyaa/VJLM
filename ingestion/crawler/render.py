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
LINKS = os.path.join(ROOT, "ingestion", "state", "rendered_links.json")
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
# Load event: DOM ready, then wait_for_stable_text() decides when the app has
# painted. "networkidle" waited for NO traffic for 500 ms, and this site never
# goes quiet - analytics beacons and the chat widget keep polling - so it cost
# 19 of 32 s on a department page. Measured on four pages: 1.0-1.8 s instead of
# 1.9-12.4 s, with byte-identical extracted text on every one (INGEST-007).
LOAD_EVENT = "domcontentloaded"
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

# Hidden content. A React page only puts the ACTIVE tab's panel in the DOM, and
# a closed accordion's body often is not there at all, so reading the page as
# loaded misses it: /academics/regulations showed only its B.Tech tab, and each
# department page only one programme's PEOs/PSOs/POs. Every tab, collapsed
# section and "show more" button is clicked, and the text of every state is
# kept. Controls inside header/nav/footer are left alone - they are the site
# menu, not content. Capped so a pathological page cannot stall the crawl.
EXPAND_MAX_CLICKS = 120
EXPAND_SETTLE_MS = 350
# Plain <button>s count too: /academics/regulations switches between B.Tech,
# M.Tech and MCA with bare buttons that carry no role or aria state, so a
# selector for "tabs and accordions" alone never opened M.Tech or MCA. What is
# excluded is anything that ACTS rather than reveals - form submits and buttons
# labelled like an action - so the crawler never submits, pays, or signs in.
CANDIDATES_JS = r"""() => {
  const sel = '[role=tab], [aria-expanded="false"], details:not([open]) > summary, button';
  const action = /\b(submit|send|log ?in|sign ?(in|up)|register|apply|pay|donate|delete|remove|log ?out|subscribe|download|print|share|copy|read aloud|accessibility)\b/i;
  const seen = {}, out = [];
  for (const e of document.querySelectorAll(sel)) {
    if (e.closest('header, nav, footer, form, [role=navigation], [role=banner], [role=dialog]')) continue;
    if (!(e.offsetWidth || e.offsetHeight) || e.disabled) continue;
    const label = (e.innerText || e.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim().slice(0, 80);
    // e.type is 'submit' for EVERY <button> without a type attribute - the
    // HTML default - so testing it excluded all plain buttons. Only an
    // explicit submit is an action; buttons inside <form> are excluded above.
    if (e.getAttribute('type') === 'submit' || action.test(label)) continue;
    // Signature = label + which occurrence of that label. Stable across the
    // re-render that follows a navigation and back, unlike a DOM attribute.
    const key = e.tagName + '|' + label;
    seen[key] = (seen[key] || 0) + 1;
    out.push(key + '|' + seen[key]);
  }
  return out;
}"""
CLICK_JS = r"""(sig) => {
  const sel = '[role=tab], [aria-expanded="false"], details:not([open]) > summary, button';
  const seen = {};
  for (const e of document.querySelectorAll(sel)) {
    if (e.closest('header, nav, footer, form, [role=navigation], [role=banner], [role=dialog]')) continue;
    if (!(e.offsetWidth || e.offsetHeight) || e.disabled) continue;
    const label = (e.innerText || e.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim().slice(0, 80);
    const key = e.tagName + '|' + label;
    seen[key] = (seen[key] || 0) + 1;
    if (key + '|' + seen[key] === sig) { e.setAttribute('data-vjlm-x', '1'); return true; }
  }
  return false;
}"""
# A fingerprint of what the page currently shows - length and a djb2 hash of
# its text, computed in the browser. Most clicks on a big page reveal nothing
# new (an already-open tab, a carousel arrow), and copying + parsing the whole
# page after each one was the slowest part of rendering a department page.
# Now the expensive extraction runs only when the fingerprint is new.
VIEW_JS = r"""() => {
  const t = document.body ? document.body.innerText : '';
  let h = 5381;
  for (let i = 0; i < t.length; i++) h = ((h * 33) ^ t.charCodeAt(i)) >>> 0;
  return t.length + ':' + h;
}"""
LINKS_JS = "() => [...document.querySelectorAll('a[href]')].map(a => a.href)"


def was_rendered(prev):
    """True if the previous snapshot is rendered text. The shrink guard must
    compare against real content: when the SPA shell was stored instead (a
    plain-HTTP fetch, INGEST-004), its 4 KB of markup made every genuine
    render look like a collapse, and the guard rejected the fix."""
    path = os.path.join(SNAP, prev.get("snapshot") or "")
    if not prev.get("snapshot") or not os.path.exists(path):
        return False
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(64).lstrip().lower()
    return not head.startswith(("<!doctype", "<html"))


def merge_lines(texts):
    """Ordered union of the lines of several renders of one page."""
    seen, out = set(), []
    for t in texts:
        for ln in t.splitlines():
            if ln and ln not in seen:
                seen.add(ln)
                out.append(ln)
    return "\n".join(out)


def expand_and_collect(page, url):
    """Click every content control, collecting text and links from each state.
    Returns (title, text, links, clicks).

    Rescans after every click: opening a tab can reveal a nested tab set
    (programme -> PEOs / PSOs / POs), and a control that navigates re-renders
    the page on the way back. What has been tried is tracked here, by
    signature, so neither case makes it click the same thing twice or stop."""
    title, text = extract_text(page.content())
    texts, links, clicks, tried = [text], set(page.evaluate(LINKS_JS)), 0, set()
    views = {page.evaluate(VIEW_JS)}
    while clicks < EXPAND_MAX_CLICKS:
        todo = [c for c in page.evaluate(CANDIDATES_JS) if c not in tried]
        if not todo:
            break
        sig = todo[0]
        tried.add(sig)
        if not page.evaluate(CLICK_JS, sig):
            continue
        try:
            page.click('[data-vjlm-x="1"]', timeout=2000)
        except Exception:
            page.evaluate("() => document.querySelectorAll('[data-vjlm-x]')"
                          ".forEach(e => e.removeAttribute('data-vjlm-x'))")
            continue
        clicks += 1
        before = page.evaluate(VIEW_JS)
        page.wait_for_timeout(EXPAND_SETTLE_MS)
        page.evaluate("() => document.querySelectorAll('[data-vjlm-x]')"
                      ".forEach(e => e.removeAttribute('data-vjlm-x'))")
        if page.url.split("#")[0] != url.split("#")[0]:
            # The control was really a link: its target is a page of its own,
            # reached through the harvested links. Go back and carry on.
            links.add(page.url)
            try:
                page.goto(url, wait_until=LOAD_EVENT, timeout=NAV_TIMEOUT_MS)
                wait_for_stable_text(page)
            except Exception:
                break
            continue
        dialog = page.query_selector('[role=dialog]')
        view = page.evaluate(VIEW_JS)
        if view not in views:
            views.add(view)
            texts.append(extract_text(page.content())[1])
            links.update(page.evaluate(LINKS_JS))
        if dialog is not None:
            # A button may open a dialog. Read it, then close it so its
            # overlay does not swallow every click after it.
            page.keyboard.press("Escape")
            page.wait_for_timeout(150)
    return title, merge_lines(texts), links, clicks

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


def _render_serial(items, total, state, delay=1.0, headful=False):
    """One browser working through (index, row) pairs. Runs in its own
    process when render_all() shards the pages across workers."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        # Without this stage every page in the knowledge base is missing or
        # stale, and on 2026-09-30 its absence went unnoticed for days
        # (INGEST-004). Fail with the fix, not a bare traceback.
        sys.exit("render.py needs Playwright + Chromium, which are not installed. "
                 "Run: pip install -r requirements.txt && "
                 "python -m playwright install chromium")

    results, links = {}, {}
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

        for i, row in items:
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
                    page.goto(url, wait_until=LOAD_EVENT, timeout=NAV_TIMEOUT_MS)
                    settled = wait_for_stable_text(page)
                    title, text, found, clicks = expand_and_collect(page, url)
                    for h in found:
                        links.setdefault(h.split("#")[0], url)
                    if not settled:
                        print("  [%d/%d] WARN  %s still changing at timeout"
                              % (i, total, url), flush=True)
                    err = None
                    break
                except Exception as e:
                    err = e
                    if attempt + 1 < RETRIES:
                        print("  [%d/%d] retry %d/%d %s (%s)"
                              % (i, total, attempt + 1, RETRIES - 1, url,
                                 str(e).splitlines()[0][:50]), flush=True)
                        time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            if err is not None:
                # Fail open: keep whatever we already had for this page.
                results[sid] = dict(prev, status="error", last_error=str(err)[:120],
                                    last_seen=time.time())
                print("  [%d/%d] ERROR %s (%s)" % (i, total, url,
                                                   str(err).splitlines()[0][:60]), flush=True)
                continue

            if len(text) < MIN_CHARS and "page not found" in text.lower():
                # The site's own 404 ("404 Oops! Page not found"). The route is
                # referenced by the app but does not exist - /about-us, /rti.
                # A distinct status so it waits out its normal interval instead
                # of being retried as a missing page on every cycle.
                results[sid] = dict(prev, status="not_found", last_seen=time.time(),
                                    last_error="site 404 page")
                print("  [%d/%d] 404   %s" % (i, total, url), flush=True)
                continue
            if len(text) < MIN_CHARS:
                results[sid] = dict(prev, status="error", last_seen=time.time(),
                                    last_error="rendered under %d chars" % MIN_CHARS)
                print("  [%d/%d] EMPTY %s (%d chars)" % (i, total, url, len(text)), flush=True)
                continue

            # Anomaly guard: a page that suddenly renders far less than last
            # time is a partial hydration, not an edit. Keep the last known
            # good snapshot rather than overwriting knowledge with a fragment.
            was = prev.get("bytes", 0) if was_rendered(prev) else 0
            if was and len(text.encode("utf-8")) < was * MAX_SHRINK_RATIO:
                results[sid] = dict(prev, status="error", last_seen=time.time(),
                                    last_error="shrank %d -> %d bytes; kept previous"
                                               % (was, len(text.encode("utf-8"))))
                print("  [%d/%d] SHRANK %s (%d -> %d bytes, rejected)"
                      % (i, total, url, was, len(text.encode("utf-8"))), flush=True)
                continue

            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h == prev.get("content_hash"):
                results[sid] = dict(prev, status="unchanged", last_seen=time.time(),
                                    last_success=time.time())
                print("  [%d/%d] same  %s" % (i, total, url), flush=True)
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
                print("  [%d/%d] %-7s %s (%d chars, %d controls opened)"
                      % (i, total, results[sid]["status"], url, len(text), clicks), flush=True)
            time.sleep(delay)

        ctx.close()
        browser.close()
    return results, links


# Pages render in parallel. Rendering is mostly waiting - network idle, text
# settling, 350 ms after each click - so it scales with browsers, not CPU. With
# every tab opened, ~630 pages took ~10 s each one at a time: a site deploy
# meant ~1h45m of re-rendering. Four workers is the same concurrency fetch.py
# already uses against the same server (INGEST-007).
RENDER_WORKERS = int(os.environ.get("VJLM_RENDER_WORKERS", "4"))


def render_all(rows, state, delay=1.0, headful=False, workers=None):
    try:
        import playwright.sync_api  # noqa: F401  - fail here, with the fix, not in a worker
    except ImportError:
        sys.exit("render.py needs Playwright + Chromium, which are not installed. "
                 "Run: pip install -r requirements.txt && "
                 "python -m playwright install chromium")
    items = list(enumerate(rows, 1))
    workers = max(1, min(workers or RENDER_WORKERS, len(rows)))
    if workers == 1 or headful:
        return _render_serial(items, len(rows), state, delay, headful)
    from concurrent.futures import ProcessPoolExecutor
    shards = [items[k::workers] for k in range(workers)]
    results, links = {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for r, found in ex.map(_render_serial, shards, [len(rows)] * workers,
                               [state] * workers, [delay] * workers):
            results.update(r)
            for h, src in found.items():
                links.setdefault(h, src)
    return results, links

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
    results, links = render_all(rows, state, a.delay, a.headful)
    state.update(results)

    # Every link a rendered page contains, for discover.py. Faculty profiles,
    # blog posts and anything else built from a route template exist only as
    # links in the rendered DOM - the bundle holds the template, not the URLs -
    # so this is the only place they can be found. Merged, never replaced: a
    # partial render (--only-ids) must not forget links other pages carry.
    known = {}
    if os.path.exists(LINKS):
        with open(LINKS, encoding="utf-8") as f:
            known = json.load(f)
    for h, src in links.items():
        known.setdefault(h, src)
    with open(LINKS + ".tmp", "w", encoding="utf-8") as f:
        json.dump(known, f, indent=0, sort_keys=True)
    os.replace(LINKS + ".tmp", LINKS)
    print("links seen in rendered pages: %d (%d known in total)" % (len(links), len(known)))

    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE)

    from collections import Counter
    c = Counter(v["status"] for v in results.values())
    print("\nDONE %d pages in %.1fs | %s" % (len(rows), time.time() - t0, dict(c)))


if __name__ == "__main__":
    main()
