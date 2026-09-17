# VJLM Decision Log

Every major technical choice is recorded here **before** the stage is frozen. A decision without measured evidence or a documented engineering trade-off is not a decision — it is a guess.

## Rules

1. Define the requirement and its hard constraints first.
2. Research current **primary** sources with the browser: official model cards, official docs, original papers, the official VNRVJIET site. Not blog comparisons.
3. Build the simplest correct baseline so the rest of the system can be tested.
4. Change only the component under test. Same benchmark, same hardware.
5. Measure with component-specific metrics.
6. Record the entry below: evidence, trade-offs, selected option, **rejected options and why**, revisit trigger.
7. Freeze the stage. Do not keep changing a selected component without evidence of regression.

## Entry template

```
### <ID> — <Question>
- **Status:** proposed | decided | superseded
- **Date:** YYYY-MM-DD
- **Requirements:** VJLM-specific hard constraints
- **Candidates:** shortlisted options + versions
- **Sources checked:** official URLs + date checked
- **Experiment:** dataset, hardware, parameters, exact commands
- **Metrics:** raw measured results
- **Selected:** chosen option + version
- **Why:** evidence-based reason
- **Rejected:** each option + why not
- **Revisit trigger:** what future condition would justify re-evaluation
```

ID prefixes: `FOUND-` foundation, `SCOPE-`, `DATA-`, `INGEST-`, `PREP-`, `EVAL-`, `RAG-EMBED-` / `RAG-RETR-` / `RAG-CHUNK-` / `RAG-STORE-`, `SLM-`, `FT-`, `QUANT-`, `AUTH-`, `PROMPT-`, `UI-`.

---

## FOUND-001 — Where does the VJLM repository live?

- **Status:** decided
- **Date:** 2026-09-16
- **Requirements:** Local path on the development machine with room for raw snapshots, multiple model checkpoints and several GGUF quantizations (tens of GB).
- **Sources checked:** `Get-Volume` on the target machine, 2026-09-16.
- **Metrics:** Only one fixed data volume exists — `C:` at 952.6 GB total, 522.6 GB free. A 1 GB unlettered NTFS recovery partition is the only other volume. There is no `D:` drive.
- **Selected:** `C:\VJLM`
- **Why:** A `D:` drive was requested but does not exist on this machine. `C:` has 522 GB free, which is ample for snapshots plus several 4–8 GB quantized builds.
- **Rejected:** `D:\VJLM` — no such volume.
- **Revisit trigger:** A second physical drive is added, or snapshot + model storage passes ~200 GB.

## FOUND-002 — Target hardware profile and what it constrains

- **Status:** decided
- **Date:** 2026-09-16
- **Requirements:** Record the exact target hardware before any model choice, per the Phase 0 gate.
- **Metrics:** See [`docs/hardware_profile.md`](docs/hardware_profile.md). Summary: Intel i7-12700H, 14 cores / 20 threads, **15.7 GB RAM**, Intel Iris Xe integrated graphics (no CUDA), Windows 11 64-bit, 522 GB free.
- **Selected:** CPU-only local inference. 15.7 GB system RAM is the binding constraint on model size, not disk and not compute.
- **Why:** The iGPU has no CUDA support and ~1 GB addressable VRAM, so it cannot train and is not a useful inference accelerator here. Fourteen physical cores make CPU inference viable at practical speeds. After OS and browser overhead the realistic resident budget for the model is roughly 6–8 GB, which points at a 3B–8B parameter model at 4–5 bit quantization — to be confirmed by measurement in Phase 7 and Phase 9, not assumed now.
- **Revisit trigger:** RAM upgrade, or deployment moves to a different on-premise server.

## FOUND-003 — Where does fine-tuning run?

- **Status:** decided
- **Date:** 2026-09-16
- **Requirements:** Fine-tuning needs a CUDA GPU. Runtime inference must remain local with no external LLM API.
- **Candidates:** Local CPU training; local iGPU training; Google Colab GPU; other rented GPU.
- **Selected:** **Google Colab GPU for training only.** Train on Colab, export the adapter and merged weights, download them, quantize, and serve locally on CPU.
- **Why:** No CUDA GPU exists locally. This keeps the hard constraint intact because Colab is a build-time tool, not a runtime dependency — the deployed system makes no network call to it. Only public VNRVJIET data and derived training examples are uploaded, so no private institutional data leaves the machine in Phase 1.
- **Rejected:** Local CPU fine-tuning — impractically slow for 7B-class models. Local iGPU — no CUDA, ~1 GB VRAM.
- **Constraints this imposes:** Colab sessions are time-limited and can be reclaimed, so training must checkpoint to Drive and be resumable. Colab GPU VRAM (commonly ~15 GB on the free T4) caps the practical method at LoRA/QLoRA rather than full fine-tuning — to be confirmed in Phase 8.
- **Revisit trigger:** Institutional GPU access becomes available, or a Phase 8 experiment shows LoRA/QLoRA gains are insufficient.
- **Operating arrangement (2026-09-16):** Colab is driven through the in-app browser. The project owner signs in; the assistant never handles credentials. The assistant then opens the notebook, runs cells and reads output. Implications: every training run must be reproducible from a notebook committed under `training/notebooks/`, not from ad-hoc cells typed into a session, and checkpoints must be written to Drive so a reclaimed runtime does not lose the run.
- **Runtime boundary restated:** Colab is used at Phase 8 (fine-tuning) and optionally Phase 9 (fp16 merge / GGUF conversion). It must never appear in the serving path.

## FOUND-004 — Base model family to shortlist

- **Status:** proposed — NOT decided
- **Date:** 2026-09-16
- **Requirements:** Open weights, license compatible with academic/on-premise deployment, instruct-tuned, LoRA/QLoRA-trainable on Colab, llama.cpp/GGUF-convertible, runs in the RAM budget from FOUND-002.
- **Candidates:** Hugging Face open-weight instruct models in the practical local size range — Mistral and Llama families named as a starting point by the project owner, plus other current candidates to be identified by browser research.
- **Why not decided:** Per the Phase 7 gate, the base model is selected **after** the benchmark (Phase 5) and baseline RAG (Phase 6) exist, by running the held-out VJLM questions against each candidate with identical retrieved context on this hardware. Naming a model now would be exactly the popularity-driven choice the plan forbids.
- **Next step:** At Phase 7, browser-verify each candidate's official model card for license, context length, tokenizer/chat template, fine-tuning support and current llama.cpp/GGUF support. Record versions and URLs here.

---

## INGEST-001 — No-API constraint on the data pipeline

- **Status:** decided (constraint set by the project owner, 2026-09-16)
- **Requirements:** The ingestion pipeline must be **dynamic, automatic and API-free**. It keeps itself current with the public VNRVJIET website without any external service in the loop and without a human triggering it.

### Revision, 2026-09-16 — free third-party services permitted, LLM APIs not

The original entry banned all third-party services. The project owner has relaxed this: **free third-party services are acceptable; hosted LLM APIs are not.** The constraint is now tiered by *where in the system* the dependency sits, because cost is not the only thing that matters — a free call in the query path still breaks local operation.

**Tier 1 — forbidden everywhere, at any price**

Hosted LLM APIs for generation (OpenAI, Gemini, Anthropic, Groq, any hosted inference endpoint). This is the project's founding non-negotiable. It applies at build time too: using a hosted LLM to generate fine-tuning QA pairs would make the model's behaviour derive from a service VJLM cannot reproduce or defend.

**Tier 2 — forbidden at runtime regardless of free tier**

Hosted embedding and reranking APIs. The reason is structural, not budgetary: **the embedding model runs on every user query**, not only when the index is built. A hosted embedding call would mean VJLM stops answering when the internet drops, when the free tier is exhausted, or when the provider retires the endpoint — and the index would be permanently tied to a vector space we cannot regenerate. Embeddings stay local.

**Tier 3 — permitted at build time when free, as accelerators with a local fallback**

Crawl and extraction helpers (e.g. free-tier readability/extraction endpoints), sitemap and link-discovery utilities, and search-based discovery to cross-check coverage. Permitted under three conditions:

1. The pipeline must still complete **without** the service. Every such helper is an optimisation over a local path that works on its own.
2. It touches only public VNRVJIET data. Nothing private is ever sent out.
3. It is recorded here with its free-tier limits and the date checked.

**Why the fallback condition is not bureaucracy:** VJLM has to keep running after the current student team graduates. Free tiers get rate-limited, change terms, or disappear. A pipeline whose only discovery path is someone's free API becomes unmaintainable the moment that key expires, and the people left holding it will not know why the knowledge base went stale.

### What "automatic" requires

Automatic means no human step in steady state. Concretely:

1. A scheduler triggers refresh per source, at an interval matched to that source's `update_frequency` — an exam schedule is not recrawled on the same cadence as the college history page.
2. Each run reconciles **new / changed / unchanged / disappeared** without supervision.
3. State is durable: last success, ETag, normalized hash, parser version, document version per source. A run resumes rather than restarting.
4. Failures are logged and retried, and **never** delete last-known-good knowledge.
5. Index sync (upsert and retire) happens in the same run. A refresh that updates snapshots but leaves the index stale is a silent failure.

### Consequences accepted

- Discovery quality now rests entirely on the registry plus sitemap plus scoped recursion. If the site has no sitemap or a stale one, that is a real coverage risk, to be measured in Phase 3 rather than assumed away.
- JavaScript-rendered pages need a **local** headless browser (a local browser is not an API). ~~This is heavier and slower, so it stays a fallback for pages plain HTTP cannot extract.~~ **Superseded 2026-09-17 by INGEST-003:** Phase 1 recon proved plain HTTP extracts *nothing* from any page route, so the browser is the primary path for pages, not a fallback.
- All extraction is deterministic parsing, which means parser failures are our problem to detect. Hence the anomaly guards in `configs/ingestion.yaml`.

- **Revisit trigger:** The institution offers an official data feed, or measured sitemap/link coverage in Phase 3 proves inadequate for a required category.

---

## INGEST-002 — `docs.vnrvjiet.ac.in` disallows all crawlers

- **Status:** **BLOCKED — needs a decision by the project owner**
- **Date:** 2026-09-16
- **Finding:** `https://docs.vnrvjiet.ac.in/robots.txt` serves `User-agent: *` / `Disallow: /`. That subdomain hosts **1,252 of the 1,923 PDFs** referenced by the site — regulations, academic calendars, Academic Council minutes, syllabus books. It is where the majority of VJLM's authoritative knowledge lives.
- **Conflict:** `configs/ingestion.yaml` sets `respect_robots_txt: true`. Honouring that excludes about 65% of the corpus and guts the answer quality VJLM is being built for. Ignoring it means running an unauthorised crawler against institutional infrastructure.

### Options

1. **Get written permission from VNRVJIET IT / the webmaster** (recommended). This is an official, faculty-sponsored project against the institution's own public content. A short written approval — ideally naming the crawler's user-agent, the rate limit and the scope — resolves the conflict cleanly and is defensible in the project report. If they can add a crawler-specific `Allow` rule to that `robots.txt`, better still: then the pipeline stays compliant with no exception logic.
2. **Restrict Phase 1 to `www` only** — the 671 `/__l5e/` PDFs plus ~100 rendered pages. Compliant with no permission needed, and enough to build and benchmark the whole system end to end. The `docs` corpus is added later once option 1 lands.
3. **Crawl it anyway.** Not recommended and not implemented without an explicit written instruction. It is an unauthorised crawl of institutional infrastructure behind Cloudflare, and the project would have to defend it in evaluation.

- **Interim action taken:** No further requests have been made to `docs.vnrvjiet.ac.in` since the `robots.txt` was read. `follow_subdomains` remains `false`.
- **Recommendation:** Pursue option 1 in parallel while building on option 2, so the pipeline is not blocked while permission is sought. The architecture does not change between them — only the registry's host allowlist.
- **Revisit trigger:** Written permission is obtained, or the subdomain's `robots.txt` changes.

---

## INGEST-003 — How page routes are fetched, and how "changed" is decided

- **Status:** decided
- **Date:** 2026-09-17
- **Requirements:** Capture the 106 page routes without a hosted service, detect real content changes, and stay cheap enough to run unattended on every app start.
- **Sources checked:** the live site, 2026-09-17 — `https://www.vnrvjiet.ac.in/` shell, `/sitemap.xml`, and `/assets/index-C3N1iQ1M.js`.
- **Experiment:** Fetched every page route over plain HTTP; rendered the same routes with local headless Chromium; probed the JS bundle for known page copy.
- **Metrics:**
  - Plain HTTP returns a byte-identical ~4 KB shell for **every** route → 0 chunks from 106 pages.
  - Headless Chromium: **105 of 106 pages** yield content, 99 distinct, 414 KB, 462 chunks. ~4 s per page.
  - All page copy is present in the single 12.7 MB bundle (7/7 probes found).
  - Bundle-hash pre-check: **one 4 KB request** replaces 105 renders when nothing shipped.
- **Selected:** **Local headless Chromium (Playwright) as the primary page fetch path**, gated by the bundle content hash, with per-page text hashes making the final call.
- **Why:** The recon finding is decisive — plain HTTP cannot see page content at all, so "fallback" was the wrong shape. A local browser is not an external API, so INGEST-001 Tier 1/2 are untouched. The bundle filename is a Vite content hash that changes iff the app changed, giving a cheap, accurate "is it worth looking?" signal.
- **Rejected:**
  - *Parsing the JS bundle directly* — the content **is** all in there, and this would be far faster. Rejected because the bundle is minified: recovering which string belongs to which route means reverse-engineering compiled React, and it breaks on every rebuild. Revisit if the site ever ships a content API or unminified data.
  - *Hashing rendered HTML* — a Vite rebuild changes every class name and chunk hash, so this would mark all 106 pages changed on every deploy and force pointless re-embedding. Hash the extracted text instead (`configs/ingestion.yaml` already required this).
  - *Fixed post-load sleep* — a guess that is either wasteful or wrong. Replaced by polling `innerText.length` until stable.
  - *Waiting on a named element per page* — accurate but needs a selector for each of 106 heterogeneous pages, which is per-page config to maintain.
- **Revisit trigger:** The site stops being client-rendered, ships a content API, or the bundle stops being content-hashed.

## RAG-CACHE-001 — What keys a cached embedding

- **Status:** decided
- **Date:** 2026-09-17
- **Requirements:** A refresh must re-embed only what changed, and must never serve a vector that does not correspond to its chunk's current text.
- **Experiment:** Replaced a chunk's text in place, leaving its position unchanged, and inspected what the cache returned.
- **Metrics:** With the cache keyed on `chunk_id` — which is `sha1(source_id:page:chunk_index)`, purely positional — the edited chunk was a cache **hit**: 0 re-embeds, cosine **1.000** against the old text and **0.413** against the new. A query for the old topic still matched the chunk at 0.731 versus 0.414 for what it now said.
- **Selected:** Key the cache on **`sha1` of the chunk text**.
- **Why:** Positional ids make "unchanged position" indistinguishable from "unchanged content", so an in-place edit silently serves a stale vector with no error anywhere. Hashing the text makes reuse mean what it should. It also collapses identical text across the corpus onto one embedding — 13,051 chunks → 11,599 distinct, 11% fewer per build.
- **Rejected:** *Keying on chunk_id* — the bug above. *Keying on (chunk_id, text hash)* — the chunk_id half adds nothing once the text is hashed, and would miss the cross-document dedup.
- **Revisit trigger:** Chunking becomes content-defined, or the embedding model changes (the cache must then be discarded, since vectors are model-specific — it is not currently namespaced by model).

## FT-001 — LoRA vs QLoRA for the VJLM fine-tune

- **Status:** decided (method); **training not yet executed**
- **Date:** 2026-09-16
- **Requirements:** Train on a free Colab T4 (~15 GB VRAM). Base model must remain swappable. Output must quantize to GGUF for 16 GB CPU serving.
- **Candidates:** full fine-tuning; LoRA; QLoRA.
- **Selected:** **LoRA**, r=16, alpha=32, dropout=0.05, on all attention and MLP projections (`q,k,v,o,gate,up,down`), base loaded in bf16.
- **Why:** At the selected 1.5B scale the base is ~3 GB in bf16 and fits the T4 with room for optimizer state and activations. QLoRA's 4-bit base load solves a VRAM constraint that does not exist here, while adding quantization error during training and reducing step throughput.
- **Rejected:**
  - *QLoRA* — buys headroom we do not need at 1.5B. **Becomes the correct choice if the base model moves to 7B/8B**, which would not fit in bf16 on a free T4.
  - *Full fine-tuning* — no measured evidence yet that it is needed, and it risks catastrophic forgetting on a 1.5B model trained on ~1.4k behavioural examples.
- **Note:** Training precision is independent of serving precision. The output is still quantized to Q4_K_M for CPU inference — that is `QUANT-001`, a separate decision.
- **Revisit trigger:** Base model changes to 7B+; or Phase 8 evaluation shows LoRA gains over base are not meaningful.
