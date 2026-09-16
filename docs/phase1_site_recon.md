# Phase 1 — Site Reconnaissance

Date: 2026-09-16. Method: `robots.txt` / `sitemap.xml` / header inspection over plain HTTPS, one rendered page via local browser, and static analysis of the site's published JS bundle. No third-party service used.

## Summary of findings

| # | Finding | Consequence |
|---|---|---|
| 1 | The site is a **client-rendered SPA** (Vite build, `<div id="root">`). Every route returns a byte-identical 4,045-byte shell. | Plain HTTP fetching retrieves **zero content**. A headless browser is the *primary* fetch path, not a fallback. |
| 2 | `sitemap.xml` lists **83 URLs, all navigation pages, zero documents**. No `<lastmod>`. | The sitemap is a weak discovery signal. It covers roughly 4% of the knowledge surface. |
| 3 | The JS bundle references **1,923 unique PDFs** — 1,252 on `docs.vnrvjiet.ac.in`, 671 on `www` under `/__l5e/assets-v1/`. | The real knowledge base is the PDF corpus, and it is ~23× larger than the entire sitemap. |
| 4 | **`docs.vnrvjiet.ac.in/robots.txt` = `User-agent: * / Disallow: /`** | **Blocker.** 65% of the document corpus sits behind a blanket crawl prohibition. See `INGEST-002`. |
| 5 | Pages return **no `ETag` and no `Last-Modified`**, with `Cache-Control: no-cache`. | Cheap change-detection hints are unavailable. The normalized-content hash is the *only* usable fingerprint. |
| 6 | `<link rel="canonical">` is hardcoded to `https://vnrvjiet.ac.in/` on **every** route. | Canonical URL cannot be read from the HTML. Use the request URL as the canonical key. |
| 7 | Apex `vnrvjiet.ac.in` 302-redirects to `www.vnrvjiet.ac.in`. Cloudflare fronts both hosts. | Normalize to the `www` host. Expect rate limiting; keep the crawl slow and polite. |
| 8 | **23 routes exist in the app but are absent from the sitemap**, including two whole departments (`/departments/csbs`, `/departments/rai`) and `/about/code-of-conduct`. | Sitemap-only discovery would silently omit real academic content. Scoped recursion is mandatory. |
| 9 | Site content is baked into the bundle; the only `/api/` reference is `/api/broadcast`. `robots.txt` on `www` disallows `/api/`. | There is no content API to depend on, which suits the no-LLM-API constraint. Content comes from the rendered DOM. |

## What this settles about the pipeline

- **Headless browser is required.** Budget for it: rendering ~100 routes is minutes, not seconds, and it must run on a schedule unattended.
- **Discovery = registry + sitemap + scoped recursion over rendered DOM.** Sitemap alone is provably insufficient (finding 8).
- **Change detection = normalized-content hash only** (finding 5). This is exactly why `configs/ingestion.yaml` pins `fingerprint: normalized_content_hash`; the site gives us no alternative.
- **`follow_subdomains: false` is wrong as written.** `docs.vnrvjiet.ac.in` holds most of the corpus — but see the blocker before changing it.
- **PDF parsing is the main extraction workload**, not HTML parsing. 1,923 PDFs versus ~100 pages.

## Open blocker

Resolution of `INGEST-002` (permission for `docs.vnrvjiet.ac.in`) is required before Phase 3 ingestion can be designed against the full corpus.
