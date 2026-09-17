"""Cheap change signal for the whole page surface.

The site is a Vite build: all 106 page routes are served by one hashed bundle,
`/assets/index-<contenthash>.js`. Vite derives that hash from the bundle's
contents, so the filename changes if and only if something in the application -
including the page copy baked into it - changed.

That gives a 4 KB question with a precise answer: "could any page have changed
since last time?" If the bundle hash is identical, no page content can differ,
and rendering 106 pages in a headless browser is pure waste.

This does NOT replace rendering. The bundle is minified, so which string
belongs to which route is not recoverable without reverse-engineering compiled
React - brittle, and it would break on every rebuild. The bundle tells us
WHETHER to render; the browser tells us WHAT the page says.

Note this is a coarse signal in one direction only: a CSS tweak or a JS
dependency bump also changes the hash, so "changed" means "worth re-rendering",
not "the knowledge changed". Per-page content hashes in render.py make the
final call, so a rebuild with no copy changes still results in zero re-indexing.
"""
import gzip, json, os, re, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, "ingestion", "state", "site_version.json")
WWW = "https://www.vnrvjiet.ac.in"
UA = "VJLM-research-crawler (VNRVJIET academic project)"


def current_bundle(timeout=45):
    """The bundle path the live shell points at, e.g. /assets/index-C3N1iQ1M.js"""
    req = urllib.request.Request(WWW + "/", headers={"User-Agent": UA,
                                                     "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    shell = raw.decode("utf-8", "replace")
    m = re.search(r'src="(/assets/index-[^"]+\.js)"', shell)
    if not m:
        # Structure changed - say so rather than silently reporting "no change"
        # and letting the knowledge base quietly go stale.
        raise RuntimeError("bundle script tag not found; site structure changed")
    return m.group(1)


def read():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f).get("bundle")
    except Exception:
        return None


def write(bundle):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"bundle": bundle}, f, indent=1)
    os.replace(tmp, STATE)


def pages_may_have_changed():
    """(changed: bool, bundle: str|None, reason: str)"""
    try:
        now = current_bundle()
    except Exception as e:
        # Fail open: if we cannot tell, assume it changed. Never skip a refresh
        # because a check failed - that is how a knowledge base goes stale
        # without anyone noticing.
        return True, None, "check failed (%s)" % str(e)[:60]
    was = read()
    if was is None:
        return True, now, "no previous bundle recorded"
    if was != now:
        return True, now, "bundle changed %s -> %s" % (was.split("-")[-1], now.split("-")[-1])
    return False, now, "bundle unchanged (%s)" % now.split("-")[-1]


if __name__ == "__main__":
    changed, bundle, why = pages_may_have_changed()
    print("pages may have changed:", changed)
    print("reason:", why)
    print("bundle:", bundle)
