"""One-off: fill in `last_modified` for sources crawled before it was recorded.

fetch.py now stores the Last-Modified header on every fetch, but the existing
sync_state was built without it, so every source is undated until its next
refresh - weeks, for monthly sources. This asks for the header alone (HTTP
HEAD, no body), which is far lighter on the college's server than the ordinary
refresh, and writes the dates into the same state file.

Only sources missing a date are requested, so re-running it costs nothing.

    python ingestion/crawler/backfill_dates.py [--limit N] [--workers 3]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch import STATE, UA, http_date  # noqa: E402

DELAY = 0.7


def head_date(url):
    req = urllib.request.Request(quote(url, safe=":/?&=%#@!$'()*+,;~"),
                                 headers={"User-Agent": UA}, method="HEAD")
    try:
        time.sleep(DELAY)
        with urllib.request.urlopen(req, timeout=30) as r:
            return http_date(r.headers.get("Last-Modified")), None
    except urllib.error.HTTPError as e:
        return None, "HTTP %s" % e.code
    except Exception as e:
        return None, str(e)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    # www.vnrvjiet.ac.in serves its pages without a Last-Modified header, so
    # asking it is 1,605 requests for nothing; docs.vnrvjiet.ac.in, which
    # holds the PDFs, answers with one.
    ap.add_argument("--host", default="docs.vnrvjiet.ac.in",
                    help="only sources on this host ('' for all)")
    a = ap.parse_args()

    with open(STATE, encoding="utf-8") as f:
        state = json.load(f)
    todo = [(sid, v["url"]) for sid, v in state.items()
            if v.get("url") and not v.get("last_modified")
            and (not a.host or urlparse(v["url"]).netloc == a.host)]
    if a.limit:
        todo = todo[:a.limit]
    print("%d sources need a date (of %d)" % (len(todo), len(state)), flush=True)

    done = failed = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(head_date, url): sid for sid, url in todo}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            sid = futs[fut]
            date, err = fut.result()
            if date:
                state[sid]["last_modified"] = date
                done += 1
            else:
                failed += 1
            if n % 100 == 0:
                print("  %d/%d  dated=%d  no-date=%d" % (n, len(todo), done, failed), flush=True)

    # Re-read before writing and merge in only the dates. The refresh cycle
    # writes this same file while we are asking for headers, so saving the
    # copy we loaded minutes ago would silently roll back its work.
    with open(STATE, encoding="utf-8") as f:
        current = json.load(f)
    for sid, v in state.items():
        if v.get("last_modified") and sid in current:
            current[sid]["last_modified"] = v["last_modified"]
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=1, sort_keys=True)
    os.replace(tmp, STATE)
    dates = sorted(v["last_modified"] for v in current.values() if v.get("last_modified"))
    print("dated %d, no header from %d -> %s" % (done, failed, STATE))
    if dates:
        print("oldest %s   newest %s" % (dates[0], dates[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
