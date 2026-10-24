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

> **Correction, 2026-09-18 — this entry describes a different machine.** The repository lives at `C:\Users\E028.10\code\VJLM`, not `C:\VJLM`. The disk figures above (952.6 GB total, 522.6 GB free) are not this machine either: `C:` here is **474.4 GB total with 43.6 GB free**. The revisit trigger has therefore already fired in substance — model storage is the binding constraint, not a distant one. `configs/paths.yaml` has been corrected, though nothing reads it (see the note in that file).

## FOUND-002 — Target hardware profile and what it constrains

- **Status:** decided
- **Date:** 2026-09-16
- **Requirements:** Record the exact target hardware before any model choice, per the Phase 0 gate.
- **Metrics:** See [`docs/hardware_profile.md`](docs/hardware_profile.md). Summary: Intel i7-12700H, 14 cores / 20 threads, **15.7 GB RAM**, Intel Iris Xe integrated graphics (no CUDA), Windows 11 64-bit, 522 GB free.
- **Selected:** CPU-only local inference. 15.7 GB system RAM is the binding constraint on model size, not disk and not compute.
- **Why:** The iGPU has no CUDA support and ~1 GB addressable VRAM, so it cannot train and is not a useful inference accelerator here. Fourteen physical cores make CPU inference viable at practical speeds. After OS and browser overhead the realistic resident budget for the model is roughly 6–8 GB, which points at a 3B–8B parameter model at 4–5 bit quantization — to be confirmed by measurement in Phase 7 and Phase 9, not assumed now.
- **Revisit trigger:** RAM upgrade, or deployment moves to a different on-premise server.

> **Superseded, 2026-09-18 — both revisit conditions fired at once.** The hardware described above is a different machine. This one is an **Intel Core Ultra 7 265, 20 cores, 31.5 GB RAM, with an NVIDIA RTX 2000 Ada (16 GB VRAM, SM 8.9, native bf16)**. The conclusion *"CPU-only local inference; RAM is the binding constraint; budget 6–8 GB for the model; the iGPU cannot train"* was correct for that machine and is wrong here in every particular. Current profile and the constraints that actually bind — **disk at 43.6 GB free, and VRAM logits rather than weights** — are in [`docs/hardware_profile.md`](docs/hardware_profile.md).

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

> **Superseded, 2026-09-18 — Colab has left the loop entirely.** This entry opens with *"No CUDA GPU exists locally."* That is no longer true: see `FOUND-002`'s correction. Phase 8 fine-tuning runs **locally** via `training/train_local.py` and `training/run_matrix.py`, and Phase 9's merge and GGUF conversion will too.
>
> Three consequences worth keeping rather than deleting:
> - The **no-external-runtime constraint was never at risk** and is now trivially satisfied — the build path no longer touches any third party.
> - The **"training must checkpoint and be resumable"** requirement survives the move, for a different reason. Colab reclaimed runtimes; here, a run killed with the session lost 48 minutes and left nothing on disk. `FT-004` records the fix.
> - The notebooks under `training/notebooks/` are now **dead code for this machine** — they target `/content/` paths, Drive checkpoints and a free T4. They are kept as the record of the Colab-era approach, not as a runnable path.

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

- **Status:** **superseded** 2026-09-18 by `SLM-001` (base + method) and `FT-003` (measurement representation). Retained in full below, because the log records what was believed and why, not only what survived.
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

### Superseded because both premises failed, 2026-09-18

Not because the reasoning was wrong — because two facts it rested on stopped being true.

1. **The T4 left the loop.** `FOUND-003` sent training to Colab because no local CUDA GPU existed. One now does: RTX 2000 Ada, 16 GB, SM 8.9, `is_bf16_supported() == True`, verified on this machine. Every VRAM argument in this entry is denominated in a free T4's ~15 GB and no longer describes the hardware.
2. **The 1.5B premise went with it.** This entry selects LoRA *at the selected 1.5B scale*. Qwen2.5-1.5B was never the selected model — `experiments/PLAN.md` designates it the **control arm**, which exists to prove the training path end to end. `FOUND-004` still has the base model as `proposed — NOT decided`. So "LoRA is right at 1.5B" answered a question nobody had settled.

Its own revisit trigger fired on both counts: *"Base model changes to 7B+"*. `SLM-001` replaces it with a measured comparison at 3B and 7B rather than an argument from a VRAM budget that no longer applies.

One judgement in it did survive contact and is worth keeping: **training precision is independent of serving precision.** `FT-003` extends that from a note into the measurement design.

## FT-002 — How the training pair is built, masked and tokenized

- **Status:** decided
- **Date:** 2026-09-18
- **Requirements:** Supervised fine-tuning must put loss on the **assistant answer only** — system, retrieved context, question and padding must all be ignored. The training prompt must be byte-identical to what `rag/retrieve.py` sends at serving time, or the model is tuned for a prompt shape it will never see. Whatever is chosen must be identical across all arms and recorded.
- **Experiment:** `training/check_template.py` — 12 assertions per model covering loss masking, BOS accounting, pad/EOS distinctness, prefix stability and serving parity. Run against all three models before any training started, per the Phase 8 gate *"do not start training until this is confirmed."*
- **Metrics:** 12/12 passed on all three, but only after two silent bugs were found and fixed. **Neither was visible on the Qwen control**, which is the finding that matters.

  | Model | BOS with defaults | pad token | prompt-prefix stable |
  |---|---|---|---|
  | Qwen2.5-1.5B (control) | **none** — no BOS token at all | `<\|endoftext\|>` (151643) | yes |
  | Mistral-7B-Instruct-v0.3 | **2** | `<unk>` (0) | **no** |
  | Llama-3.2-3B-Instruct | **2** | `<\|finetune_right_pad_id\|>` (128004) | yes |

  1. **Double BOS.** `apply_chat_template(..., tokenize=False)` emits a literal `<s>` / `<|begin_of_text|>`, and the tokenizer's default `add_special_tokens=True` then prepends BOS again. Measured: 2 with the default, 1 with `add_special_tokens=False`. Qwen2.5 has no BOS, so the control could not have exposed it.
  2. **Mistral relocates the system message.** Its template injects the system turn next to the **last** user turn, so appending an assistant turn moves it and the full-conversation render stops being a continuation of the prompt-only render — measured divergence at character 10. Rendering prompt and completion together therefore yields a **misaligned completion mask**, and trl only logs a warning.

- **Selected:** **Construct the pair, do not scrape it**, and hand trl a pre-tokenized dataset (`input_ids` + `completion_mask`), which makes it skip its own preparation entirely.

  ```
  prompt     = apply_chat_template([system, user], add_generation_prompt=True)
  completion = assistant_answer + eos_token
  labels     = [-100] * len(prompt_ids) + completion_ids
  ```

- **Why:** Labels are **built**, never derived from the input ids. That is what makes the standing constraint — *never mask labels via `input_ids == pad_token_id`, because when PAD and EOS share an id that silently masks genuine EOS targets* — structurally satisfied rather than merely observed: a real EOS inside the completion is trainable whatever the pad id is. The construction is also prefix-stable by definition for every model, including Mistral, and is byte-identical to the serving prompt.
- **Rejected:**
  - *trl's standard prompt-completion path* — tokenizes with the default `add_special_tokens=True`, producing the double BOS above.
  - *trl's conversational path* — templates prompt and completion together to find the boundary, which is exactly the Mistral failure. A warning, not an error.
  - *`assistant_only_loss=True`* — depends on the template exposing assistant spans reliably across three different families; the failure mode is silent and family-specific.
  - *Masking on `input_ids == pad_token_id`* — see above. Unsafe wherever PAD and EOS coincide, and it is not obvious from the call site that they do.
  - *Falling back to `eos_token` as pad* — avoided on all three models; each has a distinct pad. The fallback path remains in `pick_pad()` and prints a warning, because a future model may not.
- **Also fixed here:** `pick_pad()` initially accepted Llama's `<|finetune_right_pad_id|>` **inside Mistral's vocabulary**, because `convert_tokens_to_ids` returns the `<unk>` id for tokens that are not present rather than failing. A round-trip check (`convert_ids_to_tokens(tid) == cand`) is now required.
- **Revisit trigger:** A new base-model family enters the matrix, or `transformers`/`trl` cross a major version — both change template behaviour. Re-run `check_template.py` before trusting any run; the pins are recorded in `requirements-train.txt`.

## FT-003 — Why Phase 8 is not measured in the deployment configuration

- **Status:** decided
- **Date:** 2026-09-18
- **Requirements:** Measure whether fine-tuning changed behaviour, for five configurations, on one 16 GB GPU. The comparison must not systematically favour "fine-tuning did not help."
- **Candidates:** merged Q4_K_M GGUF (the deployment configuration); uniform bf16; Q8_0 GGUF; **NF4 base with an unmerged bf16 adapter**.
- **Selected:** **4-bit NF4 base, LoRA adapter attached unmerged in bf16, evaluated in-process via transformers** — for all five configurations, including the untouched base runs.
- **Why:** A LoRA delta is small next to the weight magnitude it is added to. Q4_K_M carries ~4.5 bits with per-block scales, so a merged delta below the quantization step **rounds away entirely**. That is a floor effect pushing every arm toward a null result — and a null caused by the measurement is indistinguishable from a true null, which is the single most expensive outcome available here because it looks like a finding. Keeping the adapter unmerged means the trained delta is never quantized at all.

  The obvious objection — *"both sides get the same quantization, so it controls out"* — is wrong in a specific way. It controls the **main effect** but not a (model × quant) **interaction**: quantization damage is not uniform across architectures, so the between-arm ranking can still skew even when each arm's own base/tuned pair is matched.

  This choice also fixes a confound that would otherwise break the shared baseline. Arm A trains in bf16 and arm C trains in NF4, but **both are scored on an NF4 base**. Holding inference representation constant while training representation varies is what makes one untouched Llama-3.2-3B score a valid baseline for both arms — base → tuned then isolates the adapter, and nothing else.
- **Rejected:**
  - *Merged Q4_K_M* — the floor effect above. It also duplicates `QUANT-001`, which `experiments/PLAN.md` already scopes as an f16-referenced sweep on the best tuned model.
  - *Uniform bf16* — 7B in bf16 is ~15.4 GB against 16.0 GB. On Windows/WDDM that does not OOM, it spills to host memory and generation crawls, so the timings would stop being comparable.
  - *Q8_0 GGUF* — near-lossless and would reuse the HTTP path unchanged. Held as the **fallback if bitsandbytes generation proves flaky**, not chosen, because it needs a conversion step per arm and puts llama.cpp's version in the loop of a measurement that does not otherwise depend on it.
- **Scope limit, stated rather than buried:** this measures behaviour **in a configuration that is not what gets deployed**. It does not establish that the shipped Q4_K_M build retains the gain.
- **Revisit trigger:** `QUANT-001` in Phase 9 re-tests the best arm against an f16 reference. **If the gain measured here vanishes at Q4_K_M, that is a quantization finding, not a fine-tuning one**, and the deployment build must change — not this entry's conclusion.

## SLM-001 — Which base model and adaptation method for VJLM

- **Status:** **proposed — NOT decided.** Four of five rows are in; arm C is still training. Recorded before the numbers arrived, so the design could not be retrofitted to them. It stays `proposed` after arm C lands too — see the scope limit at the bottom.
- **Date:** 2026-09-18
- **Requirements:** Open weights; licence compatible with academic/on-premise deployment; instruct-tuned; LoRA/QLoRA-trainable on one 16 GB card; GGUF-convertible for CPU serving; fits the RAM budget in `FOUND-002`.
- **Candidates / arms:**

  | Arm | Base | Training method |
  |---|---|---|
  | A | Llama-3.2-3B-Instruct | LoRA r=16, bf16 base |
  | B | Mistral-7B-Instruct-v0.3 | QLoRA r=16, NF4 base |
  | C | Llama-3.2-3B-Instruct | QLoRA r=16, NF4 base |

  Five unique configurations, not six: **A and C share one untouched Llama-3.2-3B base, measured once** and reused as the baseline for both. Arm C exists because A↔B changes model *and* method simultaneously, which violates the one-variable rule in `experiments/README.md`.
- **What each comparison licenses:**
  - base → A: the effect of LoRA. base → C: the effect of QLoRA.
  - **A ↔ C: LoRA vs QLoRA on an identical base.** Everything else is locked — examples, order, seed 42, epochs, lr, scheduler, warmup, rank, alpha, dropout, target modules, batch and effective batch, max length, chat template. Only the base-weight representation differs.
  - **C ↔ B: Llama-3.2-3B vs Mistral-7B under a matched QLoRA recipe.** This does **not** isolate parameter count — model family, architecture, tokenizer, vocabulary and pretraining all differ too, and the writeup must not claim otherwise.
- **Experiment:** `training/run_matrix.py`, 2 epochs, effective batch 16 (2×8), max_length 1280, lr 2e-4 cosine, target modules `q,k,v,o,gate,up,down` (explicit, never left to PEFT). Scored by `evaluation/eval_generation.py --backend hf` per `FT-003`, generation locked at `temperature=0, max_new_tokens=320, seed 42`. Tokenization and masking per `FT-002`.
- **Metrics:** **four of five rows recorded.** Arm C is left blank rather than estimated.

  | Arm | Role | run_id | grounded_f1 (CI95) | abstain_recall | false_abstain | citation_rate | train min | s/answer |
  |---|---|---|---|---|---|---|---|---|
  | B | base | `e38613e2` | 0.4401 (0.3952–0.4857) | **1.000** | 14 | 0.3605 | — | 22.19 |
  | B | tuned | `baffeb06` | **0.7654** (0.7192–0.8073) | **1.000** | **0** | 0.5102 | 230.6 | 20.16 |
  | AC | base | `b48d1ca6` | 0.2369 (0.2023–0.2733) | 0.283 | 12 | 0.0204 | — | 5.83 |
  | A | tuned | `822c92ed` | **0.7567** (0.7166–0.7959) | **1.000** | **0** | 0.5102 | **74.0** | **13.72** |
  | C | tuned | `0868f01f` | **0.7567** (0.7121–0.7964) | **1.000** | **0** | 0.5170 | 105.1 | **13.24** |

  All five rows are n=147 requested, 147 ok, 0 failed. `grounded_f1` CIs are bootstrap 95% over the 87 answerable items. Arm C required an allocator fix to run at all (`FT-005`); the fix changed no hyperparameter, so A↔C remains controlled.

- **`citation_rate` has a structural ceiling of 0.5918, and must never be read as if 1.0 were achievable.** Its denominator is all 147 items, but only 87 are answerable and an abstention cannot cite; 87/147 = 0.5918. Both tuned arms sit at 0.5102 = **75 of 87 answerable items cited, 86% of what is reachable.** The two arms cite 75 items each and agree on 74 of them, which is convergence on the same behaviour, not a copied number. Mistral *base* also attached citations to **10 items it abstained on**; both tuned arms do that zero times.

- **Established before any tuning, from the base rows alone:** Mistral-7B base scores **abstain_recall 1.000** — it refuses all 60 out-of-scope questions unprompted. **That metric is saturated and cannot improve.** The headroom this fine-tune can actually address is the **14 false abstentions** (refusing when the answer *was* in context) and the **36% citation rate**. Any writeup that reports a strong abstention number as evidence the fine-tune worked would be reporting a property of the base model.
- **What four rows already show — read with the A↔C caveat, since arm C is not in yet:**

  1. **The two base models are not remotely comparable, and the tuned ones nearly are.** Llama-3.2-3B base is *far* worse than Mistral-7B base on every axis: grounded F1 0.2369 vs 0.4401, citation 0.0204 vs 0.3605, and critically `abstain_recall` **0.283 vs 1.000** — it answers 43 of the 60 questions it should refuse. After training, Llama-3B LoRA reaches **0.7567 (0.7166–0.7959)** against Mistral-7B QLoRA's **0.7654 (0.7192–0.8073)**. Those intervals overlap across almost their entire width; on this evaluation the 3B model is **not measurably worse than the 7B one.**
  2. **The cost difference is not marginal.** Arm A trained in **74 minutes** against arm B's **230.6**, and generates at **13.72 s/answer** against **20.16** — 3.1× cheaper to train and 1.5× faster to score, for a difference in grounded F1 of 0.0087, which is well inside the ~0.03 noise floor this entry pre-registered. If that survives arm C and Phase 5, the 7B model is paying 3× the training cost for nothing measurable.
  3. **Fine-tuning mattered more than base-model choice.** The gap between the two *bases* is 0.2032 F1; the gap between the two *tuned* arms is 0.0087. Both arms converge on `abstain_recall` 1.000, `false_abstain` 0 and the same 75 cited items. The behaviour being taught is apparently reachable from either starting point — which is exactly the claim the three-way separation rests on, and the first direct evidence for it.

  **What none of this licenses yet:** a selection. These are templated items auto-derived from the corpus — see the scope limit below.

- **A↔C, the controlled comparison this entry was built for: LoRA and QLoRA are behaviourally indistinguishable here.**

  | | A — LoRA (bf16 base) | C — QLoRA (NF4 base) |
  |---|---|---|
  | grounded_f1 | 0.756694 | 0.756702 |
  | abstain_recall / precision | 1.000 / 1.000 | 1.000 / 1.000 |
  | false_abstain | 0 | 0 |
  | citation_rate | 0.5102 (75/87) | 0.5170 (76/87) |
  | **peak VRAM** | 10.61 GB | **4.88 GB** |
  | **train minutes** | **74.0** | 105.1 |
  | s / answer | 13.72 | 13.24 |

  The two means agree to **eight parts in a million**, which is closer than this evaluation can resolve — and closer than it looks. It is not a duplicated row: different `run_id`, and **67 of 87 items score differently**; they happen to average to the same place. 75 of 147 predictions are byte-identical, almost all of them abstentions, which are formulaic by design.

  **The trade is VRAM against wall-clock, not quality.** QLoRA used **54% less VRAM** and took **42% longer to train** — NF4 weights must be dequantized on every forward and again in backward under gradient checkpointing, and that compute is not free. Quantizing the base cost nothing measurable in behaviour on this benchmark.

  **Scope:** one model, one rank, one dataset, 87 scorable items. This says QLoRA is a sound choice *here*; it does not establish that 4-bit training is free in general, and a difference below ~0.03 F1 is under this evaluation's pre-registered noise floor either way.

- **Confirmed prediction — VRAM is bound by logits, not weights.** Arm A (3B, bf16) peaked at **10.61 GB** while arm B (7B, NF4) peaked at **7.18 GB**. The smaller model used 48% more VRAM, because Llama-3.2-3B's 128,256-token vocabulary against Mistral-v0.3's 32,768 dominates the cross-entropy tensor. `docs/hardware_profile.md` predicted this before either arm ran.

- **Statistical power, acknowledged up front:** 60 of 147 test items are abstention cases, so `grounded_f1` is computed over **87 items** across four or more comparisons. Differences under ~0.03 F1 are noise. `grounded_f1` is pre-registered as primary with bootstrap 95% CIs from per-item scores; the rest are descriptive. `mean_answer_tokens` is reported alongside, because unigram token-set F1 rewards verbosity — **a gain that disappears once answer length is matched is not a gain.**
- **Scope limit:** `evaluation/questions.jsonl` is still empty, so this is scored on 147 held-out **templated** examples. It measures grounded-answering *behaviour*, not real answer quality for real student questions — the same flattering-proxy weakness `HANDOFF.md` §3.2 records for retrieval. Phase 5 remains the blocker. Throughout, the language is **"best-performing model on the Phase 8 templated behavioural evaluation"**, never "winner".
- **Licence note for the Phase 7 criterion:** Mistral-7B-Instruct-v0.3 is **Apache-2.0** and ungated. Llama-3.2-3B-Instruct is gated behind a **custom** Llama 3.2 Community License with a 700M-MAU clause and a "Built with Llama" attribution requirement. If VJLM may ever be distributed, that is a live constraint and `Qwen2.5-3B-Instruct` (Apache-2.0, ungated) is the drop-in substitute for arms A and C.
- **Revisit trigger:** Phase 5 produces a real question set — this entry must then be re-measured, not merely cited. Also if `QUANT-001` shows the gain does not survive Q4_K_M (see `FT-003`), or if a materially better open model in the 3B–8B range appears.
- **Supersedes:** `FT-001`, and answers the question `FOUND-004` deferred.

## FT-004 — How a multi-hour training run survives

- **Status:** decided
- **Date:** 2026-09-18
- **Requirements:** One training arm runs ~3.5 h and the full Phase 8 matrix ~11 h. It must survive the lifetime of whatever started it, and an interruption must not cost the whole run.
- **Experiment:** Not designed — **observed.** Arm B was first launched as a background shell owned by the editor session. The session exited and training died at **step 34 of 150**, 47 minutes 51 seconds in, leaving an empty adapter directory. `save_strategy="epoch"` put the first checkpoint at step 75, so nothing was recoverable.
- **Metrics:** Mistral-7B QLoRA steady-state throughput is **~84 s/optimizer step** at effective batch 16 (measured across two independent launches: 83.4–85.5 s/it). The Phase 8 plan had budgeted arm B at 36 min for 3 epochs, which implies **9.6 s/step** — an **~8.7× underestimate**, and it was inferred rather than measured.
- **Selected:** three changes, all of which follow from that one number.
  1. **Detached launch.** `training/run_matrix.py` is started through WMI `Win32_Process.Create`, which parents it to `WmiPrvSE` rather than the calling shell, so it sits outside any job object that a session teardown would close. Verified by inspecting `ParentProcessId` after launch.
  2. **`save_steps=25`, not `save_strategy="epoch"`.** Bounds the cost of any kill to ~35 min instead of ~1h45m.
  3. **`--resume`** (**broken — see the correction below**), resuming from the newest `checkpoint-*` in the adapter directory, with every matrix stage idempotent — a training stage is complete when `run.json` exists, a scoring stage when its `(arm, role)` row is already in `generation_results.jsonl`. Re-running the driver after any interruption continues rather than restarting.
- **Why:** The Phase 8 plan explicitly rejected resume logic: *"Skip `--resume`: 225 steps is ~36 min for the longest arm, and `save_strategy="epoch"` + `save_total_limit=1` means restarting is cheaper than the code."* That reasoning was sound **given its premise** and wrong because the premise was an unmeasured estimate. At 9.6 s/step, restarting genuinely is cheaper than the code. At 84 s/step it is not, and the difference is the whole argument.

  This is the specific failure mode `experiments/README.md` is built to prevent — a number that was *assumed*, propagated into a design decision, and only surfaced when the decision it justified failed. It cost 48 minutes, which is cheap for the lesson.
- **Also changed:** `save_total_limit` removed entirely, and `eval_strategy="steps"` with `eval_steps=25`. Per-epoch evaluation yields 2 points per arm, which cannot distinguish "converged" from "stopped". Six points can, and keeping every checkpoint is what makes that curve **actionable** rather than merely descriptive — if validation loss turns, the earlier checkpoint still exists. Cost is ~2.5 GB per 7B arm against 43.6 GB free, and ~12% added wall-clock for the extra eval passes.

  This paid for itself on the first run it was active for. Arm B's full validation curve, now complete:

  | step | epoch | eval_loss |
  |---|---|---|
  | 25 | 0.33 | 0.2728 |
  | 50 | 0.67 | 0.2376 |
  | 75 | 1.00 | **0.2349 — best** |
  | 100 | 1.33 | 0.2627 |
  | 125 | 1.67 | 0.2706 |
  | 150 | 2.00 | 0.2754 |

  It bottoms at **step 75, exactly one epoch**, and rises monotonically for the whole of epoch 2. Three consecutive rising points is not noise — the second epoch is memorisation, and final `train_loss` of **0.1443** against `eval_loss` of **0.2754** says the same thing. Under the original per-epoch / `save_total_limit=1` configuration this would have been invisible: two eval points, one checkpoint, and the over-fit final adapter scored as though it were the trained model. The cost of seeing it was ~12% wall-clock and 2.5 GB of disk.
- **Rejected:**
  - *Windows Scheduled Task* — also fully detaches, and is the fallback if WMI ever proves unreliable. Not chosen because it leaves a registered task to clean up and needs no scheduling semantics here; this is one run, now.
  - *`Start-Process` alone* — does not reliably escape a job object with `KILL_ON_JOB_CLOSE`, which is exactly the failure being fixed.
  - *Keeping the harness-tracked background shell* — simplest, and the thing that failed. Acceptable only with resume in place, and even then it wastes up to `save_steps` of work on every session end.
- **Correction, 2026-09-19 — `--resume` does not work in this environment, and never did.** Point 3 above is **false as shipped**. Resuming raises before training starts:

  ```
  ValueError: Due to a serious vulnerability issue in `torch.load`, even with
  `weights_only=True`, we now require users to upgrade torch to at least v2.6
  ```

  `transformers` 4.56.2 gates `torch.load` behind torch ≥ 2.6 because of CVE-2025-32434, and the pinned environment is torch **2.5.1+cu121**. Model weights are unaffected — they are safetensors — but the **optimizer state is not**, and `_load_optimizer_and_scheduler` refuses to read `optimizer.pt`. There is no `save_safetensors` equivalent for optimizer state.

  **Why this went unnoticed for three runs:** the flag was passed every time and never exercised. Arm B trained straight through, so its record says `resumed_from_step: null`. Both arm C attempts logged `--resume given but no checkpoint ...; starting fresh`, because the checkpoint had been moved aside deliberately each time. The code path that mattered was only reached on the fourth attempt.

  This is the same failure this entry was written about, one level up. `FT-004` replaced an unmeasured *throughput* estimate with a measured one, and then shipped an unmeasured *recovery* claim in its place. The lesson the entry draws — that a number or behaviour which was never exercised is not evidence — applies to the entry itself. **Checkpoints every 25 steps are still real and still useful; what does not work is continuing from one.**

  Options, none yet chosen: upgrade torch to ≥ 2.6, which changes the pinned environment mid-experiment and would invalidate the comparability of anything trained after it; pin `transformers` to a version without the gate, same objection; or accept that recovery means restarting the arm and cut `save_steps` from the durability argument. The last is the honest default until Phase 8 completes, because changing the training environment between arms is a worse problem than losing a run.
- **Known cost of detaching:** the run is invisible to the launching session's task tracking, so no completion notification arrives automatically. Progress is exposed through `experiments/matrix.log` and `experiments/matrix_status.json` instead. Note that `matrix_status.json`'s `heartbeat` is written **only at stage boundaries** and is therefore not a liveness signal during a 3.5 h stage — use the log file's modification time.
- **Revisit trigger:** A training stage drops below ~20 min (a much smaller model, or a much faster card), at which point the original "restarting is cheaper than the code" argument becomes true again. Or the run moves to a machine where a scheduler exists and should own it.

## FT-005 — Why arm C (Llama-3.2-3B QLoRA) does not fit in 16 GB

- **Status:** **resolved 2026-09-19**, on the third attempt. Arm C trained 150/150 and scored (`0868f01f`). The two failed attempts and the invalid probe are kept below, because the wrong diagnoses are the substance of this entry.
- **Date:** 2026-09-19
- **What happened:** Arm C trained normally for 44 of 150 steps at ~36 s/step, completed its step-25 evaluation (`eval_loss` 0.3468) and saved `checkpoint-25`. At step 45 it stopped advancing. Thirty minutes later the step counter had not moved, while the GPU sat at **100% utilisation** and **15.92 GB of 16.38 GB VRAM**, and the process burned ~2 CPU cores. It was killed; VRAM immediately fell to 1.28 GB, confirming the training process held all of it.
- **This is the failure mode `docs/hardware_profile.md` warns about.** Windows/WDDM does not raise `OutOfMemoryError` — CUDA spills into shared host memory and the run degrades to PCIe speed instead of stopping. A step that took 36 s had not completed in 48× that. **A crash-free run is not proof it fitted, and a *hung* run is not proof of a hang.**
- **Mechanism, verified from source rather than inferred:** `peft.prepare_model_for_kbit_training` casts **every non-`Params4bit` parameter to fp32** — confirmed by reading `peft/utils/other.py` at the installed version 0.17.1. bitsandbytes leaves `lm_head` out of 4-bit quantization by default, so on Llama-3.2-3B the tied **128,256 × 3,072 embedding — 394M parameters — is upcast from bf16 to fp32**, and the logits it produces are fp32 too.

  | | vocab | head params | head in fp32 | logits @ batch 2 × 1280 |
  |---|---|---|---|---|
  | Mistral-7B-v0.3 (arm B) | 32,768 | 134M | 0.54 GB | 0.34 GB |
  | Llama-3.2-3B (arm C) | **128,256** | **394M** | **1.58 GB** | **1.31 GB** |

  Arm B survives this because its vocabulary is a quarter the size. Arm A survives it because plain LoRA never calls `prepare_model_for_kbit_training`, so its head stays bf16.
- **The counterintuitive part, stated plainly:** **QLoRA needed *more* VRAM than LoRA on the same model.** Arm A (bf16 base) peaked at 10.61 GB and finished; arm C (NF4 base) exceeded 16 GB and did not. Quantizing the base saved ~4 GB of weights and then spent more than that upcasting a very large embedding to fp32. This inverts the usual reason for reaching for QLoRA, and it is specific to *(large vocabulary, small model)*.
- **Why it ran 44 steps first:** peak memory is batch-content dependent. Examples vary in length up to the 1280-token cap, so most micro-batches are well under the worst case; steps 1–44 drew short enough pairs. This is why the failure looked like a hang partway through rather than an error at step 1, and why "it started fine" is not evidence of fit.
- **Not yet decided — every fix costs something the plan locked:** `SLM-001` locks batch, effective batch and max sequence length for A↔C precisely so the method is the only difference. Reducing any of them to make arm C fit would make A↔C a comparison of *two* variables, which is the thing this experiment exists to avoid. The candidate responses, none chosen:
  - *Skip the fp32 upcast and keep `lm_head` in bf16* — cheapest, and does not touch a locked hyperparameter, but it makes arm C's recipe differ from arm B's, weakening C↔B.
  - *Per-device batch 1 × accum 16* — holds effective batch at 16, but per-device batch is itself locked, and micro-batch size changes token-vs-sequence loss weighting under accumulation, so it is not a no-op.
  - *Rerun **both** A and C at settings that fit* — the only option that keeps A↔C strictly controlled. Costs ~74 min plus arm C's full run.
  - *Accept that the comparison cannot be made on this hardware* and report A↔C as not measured. Honest, and cheapest, but leaves `FT-001`'s central question open.
- **Attempt 1, 2026-09-19 — `--no-fp32-head`. Real, worth keeping, and insufficient.** The first option above was initially written off as "weakens C↔B". That undersold it. The fp32 upcast is not part of what A↔C compares — **arm A never had it**, because plain LoRA does not call `prepare_model_for_kbit_training`. It was a *second*, uncontrolled difference that PEFT introduced silently. Removing it for arm C leaves NF4-vs-bf16 base weights as the only difference between A and C, which is exactly what the comparison exists to isolate. **The fix makes A↔C more controlled, not less.** It is also defensible on its own terms: the upcast is there for numerical stability of *trainable* parameters and layernorms, and the head is frozen here — LoRA targets `q,k,v,o,gate,up,down` — so nothing is optimized in fp32 through it. Layernorms are left in fp32.

  Implemented as an explicit opt-in flag, off by default, so arms A and B stay reproducible from the committed file. Measured effect:

  | | first attempt | with `--no-fp32-head` | arm A, for reference |
  |---|---|---|---|
  | steady-state VRAM | **15.92 GB → thrash** | **10.57 GB** | 10.61 GB |
  | s / step | 36 → ∞ at step 45 | ~35 | ~29 |

  Arm C's *steady-state* now sits within 0.04 GB of arm A, and speed is unchanged, so the upcast was buying nothing. The step-25 `eval_loss` is **0.346857** with the bf16 head against **0.346763** with fp32 — a 1e-4 difference, which is direct evidence the change is numerically harmless.

  **It did not fix the failure.** The rerun froze at **step 44 again — the identical step**, at 16.02 GB. With `seed 42` and a fixed dataset order, step 45 draws the same micro-batches every time, so this is deterministic and batch-content driven, not random. The claim written here first — that the frozen fp32 head was "the entire difference" — **was wrong.** It was 5.3 GB of the difference, and not the part that binds.

  **Restarted from step 0, not resumed from `checkpoint-25`.** Those 25 steps were trained *with* the fp32 head; resuming would have made arm C a hybrid of two configurations and put a permanent caveat on A↔C, to save 15 minutes of an 87-minute run. The superseded checkpoint is kept as `superseded-fp32head-checkpoint-25/` rather than deleted.
- **Attempt 2 — a synthetic worst-case probe. Invalid, discarded.** A forward+backward on a full 2 × 1094 batch reported 29.74 GB peak *for the plain-LoRA config*, which cannot be reconciled with arm A's measured 10.61 GB over a complete run. The probe is wrong, not arm A: padding is **dynamic** — the collator pads to the longest sequence in each batch (`train_local.py:226`) — so a batch with both sequences at the maximum is harsher than anything the real sampler produced. Recorded because the number is misleading and someone will otherwise rediscover it.
- **What is still not explained.** Arm A cleared step 45 on the same data, same order, same seed, at 10.61 GB peak. Arm C fails there even with the head in bf16. The fp32 head accounted for 5.3 GB; **the remaining difference is not yet identified**, and no explanation is recorded here rather than a guessed one. Candidates not yet tested: bitsandbytes' transient dequantization buffers under gradient checkpointing (which recomputes, and therefore dequantizes, a second time in backward), and allocator fragmentation, which `expandable_segments` cannot mitigate on Windows.
- **Lowering `--max-length` is not available, and this is worth recording so it is not proposed again.** The cap is 1280 but the longest real example is 1094, and padding is dynamic — so the peak is set by the longest *example*, not the cap. Lowering 1280 → 1152 changes nothing at all. Lowering it enough to matter means cutting below 1094, and the comment at `train_local.py:221` already records why that is destructive: one example has a **1111-token prompt**, so a 1024 ceiling truncated its answer away entirely and left an example with no trainable token.
- **The lever that is left is the micro-batch.** Logits scale with `batch × seq × vocab`, so per-device batch **1 × accum 16** halves the dominant term while holding effective batch at 16. Cost: per-device batch is itself locked, and under gradient accumulation micro-batch size can shift per-token loss weighting — so to keep A↔C controlled, **arm A must be rerun at the same setting**, superseding `822c92ed`. That is ~74 min for A plus ~90 for C. It also widens the gap to arm B, which stays at batch 2, in a C↔B comparison already too loose for strong claims.
- **Attempt 3 — the actual cause: allocator fragmentation, not memory demand.** Instrumenting per-step `max_memory_allocated` against `max_memory_reserved` settled it in one run:

  | | allocated | reserved |
  |---|---|---|
  | steps 8–16, unfixed | 4.9–5.5 GB | 9.17 GB |
  | **step 17, unfixed** | **6.18 GB** | **13.21 GB** ← ratchet, never released |
  | **step 45, unfixed** | — | exceeded 16 GB, thrashed |

  **Arm C never needed more than ~6 GB.** The caching allocator was reserving 13.21 GB to serve 6.18 GB of live tensors — 2.14× — and ratcheting upward monotonically. Sequence lengths vary per batch, and bitsandbytes dequantizes NF4 weights into short-lived buffers on every forward *and* again in backward under gradient checkpointing, so cached blocks stop matching the next request. `expandable_segments`, the canonical fix, is **unsupported on Windows** (already tried and rejected, `docs/hardware_profile.md`).

  This also retired the "gradient checkpointing is silently inert" hypothesis: if it were, *allocated* would have been huge. Allocated was **below arm A's** 10.61 GB throughout.
- **Fix: `--alloc-guard`** — `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:256` set before torch initialises CUDA, plus `empty_cache()` each step. Measured across the crossing that killed both earlier attempts:

  | step | allocated | reserved |
  |---|---|---|
  | 44 | 4.96 GB | 10.96 GB |
  | **45** | **6.33 GB** ← heaviest batch in the run | **10.75 GB** |
  | 46 | 5.43 GB | 10.65 GB |

  Step 45 is visibly the largest batch — allocated peaks there, ~1.3 GB above its neighbours, exactly as a fixed seed predicts. Reserved did not react at all; it was *lower* than step 44's. Reserved then held in a 10.5–11.0 GB band for all 150 steps instead of climbing. Final peak allocated: **4.88 GB**, the lowest of any arm.

  **This is why arm A did not have to be rerun.** The fix is allocator configuration — it changes no hyperparameter and no arithmetic, so every value `SLM-001` locks for A↔C is untouched. The alternative under consideration, per-device batch 1 × accum 16, would have cost a ~3 h rerun of arm A *and* weakened the comparison. The 26-minute diagnostic bought both back.
- **`--vram-cap` is kept on for arm C**, so any recurrence raises `OutOfMemoryError` at a known ceiling instead of silently thrashing. On this platform that is the difference between a diagnosis and a hang.
- **Remaining cost, stated:** arm C's recipe now differs from arm B's, which kept the fp32 head harmlessly on a 32k vocab. `C↔B` was already the loose comparison — different family, tokenizer, vocabulary and pretraining — so this adds one more item to a list that already forbids strong claims. Trading precision in the loose comparison to gain it in the tight one is the right direction, but it is a trade, not a free fix.
- **Revisit trigger:** a card with more than 16 GB, or a decision from the above list. Whichever comes first.
- **What survives regardless:** arm C's failure does not touch arms A and B. The finding that tuned Llama-3.2-3B matches tuned Mistral-7B on this evaluation rests on `822c92ed` and `baffeb06`, both complete and both scored on all 147 items.

## QUANT-001 — Which GGUF build ships for CPU inference

- **Status:** **decided for both models.** Llama-3.2-3B → Q6_K + imatrix (2.462 GB). Mistral-7B-v0.3 → Q5_K_M (4.783 GB). The two needed different compression; see below.
- **Distribution:** both selected GGUF builds are published on Hugging Face, under Vaishnavi Sriyaa's account. GitHub carries the adapters they were built from plus every script and result needed to reproduce them (`HANDOFF.md` §0), not the 7.25 GB of quantized weights themselves.
- **Date:** 2026-09-21
- **Requirements:** 100% local CPU inference, no external API. Smallest build whose quality is *measurably* unchanged from full precision. Quality first, then size, then speed.
- **Tooling:** llama.cpp pinned at **b11050 / commit `60b06ab9a`** (`runtime/llama/PINNED_TAG.txt`). CPU build for every deployment number; CUDA build only at build time for the importance matrix and reference logits, never for a reported throughput figure.

### The measurement problem this entry is built around

The Phase 8 behavioural set is **87 scorable items** and `SLM-001` pre-registered a ~0.03 F1 noise floor. Quantization damage above Q5 is far smaller than that, so **that instrument cannot see the effect being measured.** Reading "F1 didn't drop" as "quantization was free" would be drawing a conclusion from a tool incapable of supporting it.

So quality is judged primarily by **KL-divergence per token against the model's own BF16 reference** — ~16,384 paired observations instead of 87 — with the behavioural set as a confirmatory gate that nothing large broke.

The results below vindicate that choice. Between Q6_K and Q5_K_M, behavioural F1 scatters ±0.005 **with no ordering at all** — Q5_K_M, which fails KLD by 2.4×, scored *+0.0048 above the reference*. Only KLD separates them.

### Three reference levels, so errors stay separable

| level | artifact | isolates | measured |
|---|---|---|---|
| **L0a** | adapter on bf16 base, unquantized | the true fine-tune at full precision | 0.7513 |
| **L0b** | merged bf16 weights | **merge error** | 0.7539 → **+0.0026** |
| **L1** | BF16 GGUF | **conversion / chat-template error** | 0.7521 → **−0.0018** |
| **L2** | quantized builds | **quantization error** | see table |

Total path error, unmerged → GGUF: **+0.0008**. Citation rate and abstention were *identical to four decimals* across all three levels, which is the evidence the chat template survived conversion — the failure that most often corrupts GGUF results silently and would have poisoned every number below it.

### Pre-registered gates, and three corrections I had to make

Registered before any build was scored. Three needed fixing, all recorded rather than quietly adjusted:

1. **`abstain_recall` stays 1.000** → **wrong baseline.** 1.000 came from Phase 8's *NF4* inference; the BF16 reference is **0.9833**. Corrected to *relative to L1*, before any quant was judged.
2. **top-1 token flips ≤ 1%** → **mis-calibrated.** Q8_0, the most conservative build on the ladder, flips **2.18%**. As written the gate disqualified everything. Demoted to a diagnostic: most flips are near-ties where either token is fine. Damage is judged from the Δp distribution's asymmetry instead. Corrected having seen only Q8_0's scale.
3. **Identical-output rate** was never a gate, and is explicitly not one — a quantized model can pick a different token in a near-tie with no quality loss.

Final gates: mean KLD ≤ 0.01 · `abstain_recall` ≥ 0.9833 · `false_abstain` = 0 · `citation_rate` ≥ 0.5038 · paired F1 delta ≥ −0.01. **Narrow miss** = fails one gate by <20% of its margin → build the imatrix variant before stepping up a size class.

### Llama-3.2-3B — results

Lineage: **`meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f`** → **LoRA r=16** (arm A, final step 150, `822c92ed`) → merged bf16 → **GGUF Q6_K + VJLM imatrix**.

| build | GB | mean KLD | f1 | Δ vs ref | cite | absR | gate |
|---|---|---|---|---|---|---|---|
| BF16 (L1 reference) | 5.992 | — | 0.7521 | — | 0.5238 | 0.9833 | — |
| Q8_0 | 3.187 | 0.001647 | 0.7526 | +0.0005 | 0.5238 | 0.9833 | **pass** |
| **Q6_K-imat** | **2.462** | **0.009661** | **0.7521** | **+0.0000** | **0.5238** | **0.9833** | **pass — SELECTED** |
| Q6_K | 2.462 | 0.010682 | 0.7550 | +0.0029 | 0.5238 | 0.9833 | fail, by 6.8% |
| Q5_K_M-embd8 | 2.252 | 0.022547 | 0.7548 | +0.0027 | 0.5238 | 0.9833 | fail |
| Q5_K_M-imat | 2.163 | 0.019743 | 0.7494 | −0.0027 | 0.5102 | 1.0000 | fail |
| Q5_K_M | 2.163 | 0.023813 | 0.7569 | +0.0048 | 0.5238 | 0.9833 | fail |
| Q4_K_M | 1.881 | 0.063531 | 0.7408 | −0.0113 | 0.5102 | 0.9667 | fail |

- **Selected: `Q6_K-imat`, 2.462 GB — 2.43× compression, and the only build that reproduced the reference exactly on grounded F1, citation rate and abstention simultaneously.** Q8_0 also passes and is 29% larger for nothing measurable; under quality → size → speed, size breaks the tie.
- **Q4_K_M is the build that validates the method.** It is the only one where behaviour visibly degrades — F1 −0.0113, citation down, and **`abstain_recall` 0.9667**, i.e. it answers two questions it should refuse. That is the hallucination axis moving. Its KLD is 38× Q8_0's. Both instruments agree at the bottom of the ladder, which is why KLD's verdict higher up is credible rather than an uninterpretable number. **Per the standing instruction, Q4 ships only if it passes the same gates. It does not.**

### The imatrix question, answered by measurement

| quant | plain | + VJLM imatrix | improvement |
|---|---|---|---|
| Q5_K_M | 0.023813 | 0.019743 | **−17.1%** |
| Q6_K | 0.010682 | **0.009661** | **−9.6%** |

It helps measurably, and **more at lower precision** — the predicted pattern, now with VJLM numbers. At Q6_K it was the difference between failing and passing: **the narrow-miss rule fired on a threshold, not a hunch, and saved 725 MB (23%) against falling back to Q8_0.**

Corpus: 518k tokens, 61% sampled `rag_chunks.jsonl` + 39% `train.jsonl` rendered in the model's own chat template, built by `quantization/make_imatrix_corpus.py` with `--parse-special`. Val and test splits are excluded — calibrating on held-out text would leak the benchmark into the artifact being benchmarked.

### Rejected, with reasons

- **Q8_0** — passes every gate, 29% larger than the selected build for no measurable quality. Kept on disk as the conservative fallback if a future corpus ever shows Q6_K-imat straining.
- **Q5_K_M and its variants** — KLD 0.0197–0.0238, two to three times the gate. Behaviourally they look fine, which is precisely the trap this entry exists to avoid.
- **`Q5_K_M-embd8`** (`--token-embedding-type q8_0 --output-tensor-type q8_0`) — the hypothesis that Llama's 128,256-token vocabulary makes embedding/output tensors worth protecting. **Tested and rejected:** 5.3% better KLD for 4% more size, and still nowhere near the gate. Real effect, not a useful one.
- **Q4_K_M** — fails on both instruments, and is the only build to break abstention.

### Mistral-7B-Instruct-v0.3 — results

Lineage: **`mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708`** → **QLoRA r=16, NF4 base** (arm B, final step 150, `baffeb06`) → merged bf16 → **GGUF Q5_K_M**.

| build | GB | mean KLD | gate |
|---|---|---|---|
| BF16 (L1 reference) | 13.5 | — | — |
| Q8_0 | 7.174 | 0.000550 | pass |
| Q6_K | 5.539 | 0.003090 | pass |
| **Q5_K_M** | **4.783** | **0.007710** | **pass — SELECTED** |
| Q4_K_M | 4.073 | 0.021360 | fail, 2.1× over |

Behavioural gate on the selected build, against its own BF16 reference:

| metric | BF16 ref | Q5_K_M | delta |
|---|---|---|---|
| grounded_f1 | 0.7678 | **0.7726** | +0.0048 |
| citation_rate | 0.5102 | 0.5102 | ±0.0000 |
| abstain_recall | **1.0000** | **1.0000** | ±0.0000 |
| abstain_precision | 1.0000 | 1.0000 | ±0.0000 |
| false_abstain | 0 | 0 | — |

- **Selected: `Q5_K_M`, 4.783 GB — 2.82× compression.** Smallest build clearing the KLD gate, and it holds *perfect* abstention: it never answers a question it should refuse.
- **Q4_K_M tested and rejected** at 0.021360, per the standing instruction that Q4 ships only on proof.

### The two models needed different compression, and that is the point

| | Llama-3.2-3B | Mistral-7B-v0.3 |
|---|---|---|
| vocabulary | 128,256 | 32,768 |
| Q5_K_M KLD | 0.023813 — **fails** | **0.007710 — passes** |
| Q6_K KLD | 0.010682 — fails | 0.003090 — passes |
| selected | **Q6_K + imatrix** | **Q5_K_M** |

**Mistral tolerates the same quant type 3.1× better.** With a vocabulary a quarter the size, the embedding and output tensors are a far smaller share of the model, so quantization damage spreads thinner. Llama needed a higher precision *and* an importance matrix to clear the gate that Mistral cleared two rungs lower.

Had Llama's answer been assumed to transfer, Mistral would have shipped at Q6_K — **0.76 GB larger than necessary**. This section was deliberately left empty until measured for exactly that reason.

### Head to head — the two deployment builds

| | **VJLM-Llama-3B** | **VJLM-Mistral-7B** |
|---|---|---|
| compression | Q6_K + imatrix | Q5_K_M |
| size | **2.462 GB** | 4.783 GB |
| mean KLD | 0.009661 | **0.007710** |
| grounded_f1 | 0.7521 | **0.7726** |
| abstain_recall | 0.9833 | **1.0000** |
| citation_rate | **0.5238** | 0.5102 |
| CPU tokens/sec | **14.70** | 8.07 |
| ~time for a 35-token answer | **~2.4 s** | ~4.3 s |

**Mistral answers better; Llama deploys better.** Mistral leads on grounded F1 by 0.0205 and never mis-refuses. Llama is 1.94× smaller and 1.82× faster. Neither dominates, so the choice depends on the deployment box: on constrained hardware Llama is the pick, and where 4.8 GB and ~4 s latency are acceptable Mistral is measurably the better answerer.

### Scope limit specific to Mistral — its evidence is thinner than Llama's

Phase 9 was time-boxed when the machine had to be vacated. **Llama got a behavioural row for all seven builds; Mistral got one, on the winner.** So the Llama-style cross-check — where Q4_K_M failed on *both* instruments and confirmed they agree — is **not** repeated here. Mistral's rejected builds have a KLD number and nothing else, and Mistral's CPU throughput was measured for the selected build only.

This does not weaken the selection, which rests on KLD plus a passed behavioural gate on the build that ships. It does mean the two models' evidence is not equally complete, and the tables above should not be read as though it were.

### Scope limit

Every number here is measured on **templated held-out items auto-derived from the VJLM corpus**, the same weakness `HANDOFF.md` §3.2 records for retrieval. They rank builds of the same model against each other on identical data, which is what this decision needs. They do **not** establish real answer quality — `evaluation/questions.jsonl` is still empty and Phase 5 remains the blocker.

**Hindi and Telugu were requested and are out of scope, with a reason.** The corpus contains **zero Devanagari** across all 1,493 train/val/test rows and **4 of 13,051** RAG chunks contain any Telugu. The models were never trained on either language and no evaluation set exists. There is nothing representative to calibrate an imatrix on and nothing to verify against, so multilingual quality is **unmeasured**, not preserved. Doing it properly needs a human-written Hindi/Telugu question set first — and it would measure the *base* models' multilingual ability, not anything the fine-tune taught.

### Revisit trigger

Phase 5 produces a real question set — re-measure, do not merely cite. Also if the deployment target changes (a different CPU changes the throughput ranking, not the quality one), or if llama.cpp's quantization changes materially from the pinned build.

---

## RAG-SIM-001 — Which similarity measure ranks VJLM's retrieval

- **Status:** **decided for keyword-style queries: `coverage`.** Provisional for real student questions — see the instrument limitation below, which is the whole point of this entry.
- **Date:** 2026-09-26
- **Design:** [`docs/similarity_design.md`](docs/similarity_design.md)
- **Requirements:** Rank candidates; **support abstention** (a measure that only ranks always has a top-1 and can never say "nothing here"); bridge vocabulary gaps; respect VNR structure (R25 ≠ R22); be explainable, because every answer cites sources.

### The measure, as suggested and as extended

The project owner's supervisor proposed a graded term ladder: **1.0 if the word is present, 0.5 if a synonym is present, 0 otherwise.** Implemented in `rag/coverage.py`, IDF-weighted so that covering a *discriminating* term counts for more than covering a common one:

```
            Σ_t  idf(t) · tier(t, d)
Cov(q, d) = ────────────────────────     ∈ [0, 1]
                 Σ_t  idf(t)
```

`students` occurs in most of the corpus and separates nothing; `condonation` occurs in a handful of documents and separates almost perfectly. An unweighted average treats them alike, which is why the IDF weighting is not decoration. Unseen query terms take the maximum observed IDF: maximally rare, unmatchable, so they only ever enter the denominator and correctly penalise every document for missing something distinctive.

Synonyms come from `rag/synonyms.py` — WordNet plus a small hand-curated domain map, gated on the indexed vocabulary.

### Measured — 90 benchmark questions, same index (13,051 chunks), 2026-09-26

| mode | Recall@5 | MRR | p50 ms |
|---|---|---|---|
| **coverage** | **0.944** | **0.836** | 35.0 |
| **coverage+syn** | **0.944** | **0.836** | 14.0 |
| lexical (0.6·BM25 + 0.4·cov) | 0.933 | 0.822 | 17.4 |
| bm25+syn | 0.944 | 0.815 | 29.6 |
| bm25 | 0.911 | 0.803 | 19.0 |
| hybrid3 (0.35/0.25/0.40) | 0.900 | 0.780 | 43.6 |
| hybrid (0.5 BM25 + 0.5 cos) | 0.789 | 0.722 | 33.5 |
| **dense** (cosine alone) | **0.667** | **0.509** | 326.0 |

- **Selected:** `coverage` — the supervisor's ladder, IDF-weighted. Best MRR and best Recall@5, and the cheapest of the strong arms.
- **Why:** It beats BM25 on both metrics (MRR 0.836 vs 0.803). BM25 is a term-*frequency* measure, so a page repeating "attendance" fifty times outranks one covering attendance *and* examinations *and* condonation. Coverage cannot be won by repetition — a term contributes its IDF once or not at all — and breadth is the better signal for the multi-concept questions students actually ask.

### Two predictions in the design document that the experiment refuted

Recorded because a decision log that only preserves correct guesses is worthless.

1. **"Cosine will be the strongest signal" (γ = 0.40). Wrong.** `dense` finished *last*, MRR 0.509 against coverage's 0.836, and 17× slower. Every arm containing cosine was dragged down in proportion to its weight: hybrid (0.5 cos) → 0.722, hybrid3 (0.40 cos) → 0.780, no cosine → 0.803–0.836.
2. **"Fusion beats its parts." Wrong here.** Every fusion lost to `coverage` alone. Mixing BM25 back in cost 0.014 MRR; mixing cosine in cost more. On this corpus one good signal beat every blend tried.

### The instrument limitation — read before quoting any number above

The 90 questions are generated by lifting ~12 words **verbatim** from the gold passage. That is not a paraphrase; it is a bag of exact keywords.

- Coverage is *defined* as exact keyword overlap, so it is being measured on precisely its strength.
- Cosine is asked to do semantic matching on a query that requires none, and pays 326 ms for the privilege.
- **The 0.5 synonym tier never fires.** `coverage` and `coverage+syn` are identical to three decimal places, because the literal word is always present. The supervisor's rule is correctly implemented and this benchmark is *structurally incapable* of testing it.

So the honest claim is narrow: **on keyword-style queries, lexical retrieval wins decisively, and coverage is the best lexical measure.** It is *not* established that embeddings fail on real student phrasing — `evaluation/paraphrase_probe.py` shows that for "what happens if I miss too many lectures" every lexical arm returns faculty CVs, and no amount of keyword weighting fixes that.

- **Rejected:** `dense` alone (worst measured, slowest); `hybrid`/`hybrid3` (fusion lost to its own components); `bm25` alone (beaten on both metrics by a cheaper measure).
- **Not yet built:** the `near` tier (0.25 for embedding-adjacent terms), metadata alignment including the **negative** term for a contradicting regulation, and the abstention threshold θ. All three are specified in the design document.
- **Revisit trigger — and this one is expected to fire:** the Phase 5 human benchmark. Paraphrased questions are the case coverage cannot handle and embeddings exist for. Until those questions exist, `dense = 0.509` should be read as "untested on the queries it was built for", not "embeddings do not work here".

### Abstention (designed, not implemented)

The reason this matters for VJLM specifically: BM25 is unbounded and pool-relative, so min-max normalising makes the best candidate score 1.0 **whether or not it is any good**. An abstention gate built on it would never fire. Coverage and cosine are both absolutely scaled — a fraction of achievable IDF mass, and a similarity on unit vectors — so the gate must be built from those:

```
Confidence(q,d) = β'·Cov(q,d) + γ'·Cos(q,d)      θ fitted on the
if max_d Confidence < θ  →  abstain              unanswerable questions
```

The evaluation set already contains the labelled negatives needed to fit θ — the "what are my internal marks" class, whose answers are genuinely absent from public data. That is what they were written for. Fitting must also check interaction with the fine-tuned model's own abstention, or the two will double-count into refusing answerable questions; `evaluation/eval_generation.py` already reports false-abstention rate.

---

## CORPUS-001 — The corpus contains almost no student-policy content

**Date:** 2026-09-30
**Status:** OPEN — this constrains what VJLM can honestly claim to be
**Found by:** the live six-technique comparison, which refused all four questions

### What happened

`evaluation/compare_modes_live.py` asked four student questions through all six
retrieval modes. **All 24 calls abstained**, including the literal question
"What is the minimum attendance required to appear for the semester end
examination?" — a rule that every Indian engineering college publishes.

Six identical rows is not a tie between techniques. It is an instrument reading
zero. So the first thing checked was whether the answer was in the corpus.

### It is not. Measured on 12,978 chunks

| probe | count |
|---|---|
| chunks containing `75%` **and** "attendance" | **0** |
| chunks mentioning "attendance" at all | **22** |
| of those 22, chunks stating a policy | **0** — all are faculty paper titles, e.g. *"Smart Attendance System Using Speech Recognition"* |
| "condonation" / "detention" | **2** / **1** |
| `fees_scholarships` category | **8 chunks** |
| `academic_calendar` category | **2 chunks** |

**The `academic_regulations` category is mislabelled.** It is the largest slice
of the corpus at 5,269 chunks, and all 32 of its files are
`II-B.Tech-*-R25.pdf` / `M.Tech-*-R25.pdf` — **syllabus and course-structure
booklets**, not the rulebook. Page 2 of `II-B.Tech-CSE-R25.pdf` is a credits
table; page 45 is a lab syllabus. Nothing in the category states a regulation.

### What VJLM currently is

Not a student handbook. On present data it is a **syllabus and course-structure
search engine with a faculty-CV database attached** — faculty is 3,841 chunks,
the second-largest category. That is why `paraphrase_probe.py` sees lexical arms
return faculty CVs for "what happens if I miss too many lectures": there is
nothing better in the index to return, so the nearest text wins by default.

### Root cause — already recorded, now shown to be load-bearing

`docs.vnrvjiet.ac.in/robots.txt` is `Disallow: /`, excluding **1,252 PDFs**
(`INGEST-002`, BLOCKED). The regulations booklet, fee structure and scholarship
policy are almost certainly among them. This is not a scraper defect and must
not be "fixed" by crawling anyway — it needs written permission from VNRVJIET
IT. `INGEST-002` is hereby promoted from a background caveat to **the single
biggest open blocker on the project**.

### Consequence for RAG-SIM-001 — sharper than the caveat already recorded

`RAG-SIM-001` notes the 90 questions are built by lifting ~12 words verbatim
from the gold passage. Given the above, those gold passages are overwhelmingly
**syllabus tables**. So `coverage = 0.944 Recall@5` substantially measures
"can you find the course-code table these twelve words were copied from" — much
closer to a lookup than to retrieval.

The number is not wrong and the ranking between lexical variants still holds.
But the claim it supports is narrower again: **coverage is the best lexical
measure for locating syllabus content by its own keywords.** Whether it is the
right measure for VJLM's intended job is unmeasured, because the content that
job requires is not in the index.

### Actions

1. **Rename the category.** `academic_regulations` → `syllabus` in the
   classifier. A citation reading "[academic_regulations]" under a credits
   table is a false claim of authority shown to the student.
2. **Ground every evaluation question in verified content.** Done for
   `compare_modes_live.py`; the header now carries the evidence for each
   question. Applies to the Phase 5 human benchmark too — writing questions the
   corpus cannot answer produces another instrument that reads zero.
3. **Request permission for `docs.vnrvjiet.ac.in`.** Until then, say plainly in
   any demo that VJLM answers curriculum and faculty questions, not policy.

### Revisit trigger

Permission granted for `docs.vnrvjiet.ac.in`. Adding ~1,252 policy PDFs would
change the corpus's composition fundamentally and **every retrieval number in
this log would need re-measuring** — coverage's advantage rests on matching
keywords in tabular syllabus text, and prose regulations are not that.

---

## INGEST-003 — OCR for image-only PDFs, and the category that lied

**Date:** 2026-09-30
**Status:** DECIDED — RapidOCR (ONNX) fallback in `preprocessing/extract.py`
**Trigger:** CORPUS-001, then a browser inspection of `www.vnrvjiet.ac.in` to
separate "the pipeline cannot extract it" from "the pipeline may not fetch it".

### What the browser inspection separated

| gap | host | robots.txt | cause | fixable here |
|---|---|---|---|---|
| Rulebooks (R25, R22, A19, R19, R18), condonation forms | `docs.*` | `Disallow: /` and explicitly `Disallow: /assets/` | policy, not technology | **no** — INGEST-002 |
| Current academic calendars (2026-27) | `www` | allowed | **scanned images, no text layer** | **yes** |
| Syllabus booklets filed as regulations | `www` | allowed | **classifier rule order** | **yes** |

Sitemap coverage was checked and is not the problem: 82 of 83 sitemap pages
are in the corpus. Discovery also finds the calendar PDFs — all 7 are in the
catalog, fetched, and snapshotted. They were lost one stage later.

### Gap 1 — 25 of 672 PDFs have no text layer

`pypdf` reads text layers. These files are one embedded image per page, so it
returned 0 characters. `extract.py` labelled them `empty (needs OCR, never
silently dropped)` and logged them — and nothing ever read the log. Four of the
five current academic calendars were among them, so "when does the semester
start" was fetched on every refresh and never answerable. 68 pages in total.

**Decision: RapidOCR, ONNX build, CPU.**

- **Rejected: Tesseract.** Needs a system binary installed outside pip; the
  pipeline would stop being reproducible from `requirements`.
- **Rejected: EasyOCR / PaddleOCR.** Pull in torch or paddle — hundreds of MB
  for 68 pages.
- **Chosen:** `rapidocr_onnxruntime` 1.4.4 runs on the same `onnxruntime`
  1.24 that fastembed already requires. Models ship inside the wheel, so nothing
  downloads at runtime. Pages rendered with `pypdfium2` at 200 DPI.

Measured on the II B.Tech calendar: 147 text lines, confidence ~0.99, **~24 s
per page on CPU**.

### The recogniser drops English spaces — repaired, and how that was chosen

The bundled model is Chinese+English and joins English words:
`CommencementofClasses`, `AcademicYear2026-2027`. To BM25 each is one token no
student types, so the text would be present and still unfindable.

- **Tried and rejected: recovering spaces from character-box gaps.** Measured
  on a line with known spacing: `Program|10|mes` has the same 10 px gap as a
  real word break. Not separable.
- **Chosen: repair from the text.** Every missing space observed sat at a case
  change, a letter/digit boundary, or a glued function word. Rules cover those;
  function-word splits (`Marksin` → `Marks in`) apply only when WordNet says
  the joined token is *not* a word and the remainder *is*, so `within` is left
  alone. Course codes (`25PC1CS205`) and regulation codes (`R25`) are
  preserved by requiring ≥3 letters on the letter side of a digit split.
- OCR boxes are regrouped into visual rows so a calendar row stays one line:
  `Commencement of Classes | 06th July 2026 | 07th December 2026`.

### Properties the unattended refresh depends on

- **Cached by SHA-256 of the PDF bytes** (`ingestion/state/ocr_cache/`). At
  ~24 s a page, re-reading unchanged scans every cycle would be unacceptable.
- **`MAX_PAGES = 30`** per document, so one large scan cannot stall a refresh
  for an hour. Truncation is recorded in the cache entry.
- **OCR is additive.** If the OCR stack is not installed, `ocr_pdf` returns
  `None` and the document stays `empty` exactly as before.
- **Chunks carry `"ocr": true`.** OCR text is a reading of pixels, not the
  publisher's characters; a citation should be able to say so.

### Gap 2 — the classifier (fixes CORPUS-001 action 1)

The first category rule matched `\br\d{2}\b`. Every syllabus booklet is named
`II-B.Tech-CSE-R25.pdf`, and every current calendar carries `-R25` too, so all
of them were filed as `academic_regulations` before the syllabus or calendar
rules were tried. Rule order is now calendar → regulations (words only) →
syllabus (which now owns the bare code). Measured effect on the catalog:
**32 syllabus booklets and 2 calendars moved, nothing else changed, no source
added or lost.** One source is now labelled a regulation — the
`/academics/regulations` page — which is the true count.

**Not fixed, deliberately:** `training/notebooks/colab_all_in_one.py` carries
the same rule. Changing it would silently change what the already-trained
adapters are evaluated against. It must be fixed before the next training run,
and the questions generated from syllabus text with the template "What does the
regulation say about…" should be regenerated with it.

### Revisit triggers

- OCR quality on a document that matters (a fee notice, an exam timetable)
  turns out wrong in a way a student would act on → add a confidence floor per
  line and surface `ocr: true` in the UI citation.
- A scanned document larger than 30 pages appears → decide per-document rather
  than raising the cap globally.

---

## RAG-IDX-001 — Dense vectors silently misaligned with the BM25 index

**Date:** 2026-09-30
**Status:** FIXED — two changes; one result set invalidated

### The defect

`rag/index.py` rebuilds BM25 first, then dense. `build_dense()` opened its
cache with `np.load(CACHE)` and never closed it. An `.npz` loads lazily and
holds its file handle, and on Windows an open file cannot be replaced — so
`save_cache()`'s `os.replace()` raised `PermissionError: Access is denied` on
**every incremental build**. `index.py` catches that, prints "dense skipped",
and carries on. Only from-scratch builds (no cache file yet) ever succeeded,
which is why the Sep 27 build worked and hid the bug.

The result: BM25 rebuilt, `dense.npy` left from an older chunk list. At 11:52
today the app's own auto-refresh did exactly this — **13,051 dense rows against
12,978 chunks**. The retriever had no check, so row *i* of the matrix (one
chunk's meaning) was paired with row *i* of the metadata (a different chunk's
text and URL). Every dense hit after the first changed document returned the
wrong passage, under a plausible citation.

### What it invalidates

- **Both live comparisons run today** (`compare_modes_live.py`, the attendance
  set and the grounded set): every `dense`, `hybrid` and `hybrid3` row. Those
  modes were scoring one chunk and returning another. HTML pages sort first and
  were stable, so page-level hits such as `icc` / `anti-ragging-cell` may have
  been aligned by luck — but that cannot be told apart from the rest, so none of
  it is evidence. The `bm25`, `coverage` and `lexical` rows are unaffected.
- **Not affected: RAG-SIM-001.** That benchmark ran immediately after the Sep 27
  from-scratch build, with 13,051 rows on both sides.

### The fixes

1. **Close the cache before writing it** (`with np.load(...) as z:` and copy
   the arrays out). Verified: the incremental build now embeds 1,289 new
   vectors, prunes 1,271 orphans, and writes a (13149, 384) matrix.
2. **The retriever refuses misaligned vectors.** If `dense.npy` rows ≠ index
   chunks, dense is disabled with a loud log line and the modes fall back to
   lexical. The principle: *a missing signal is recoverable; a wrong one is
   not.* Stale vectors produce confident, cited, wrong answers, which is the
   worst failure this system can have.

### Revisit trigger

The index is rebuilt while the app is serving and the app does not reload. The
serving process keeps its old in-memory retriever until restarted or until its
own refresher rebuilds; the guard only protects the next load. If that window
causes a real problem, index files should carry a shared build id that the
retriever checks.

---

## INGEST-002 (update) — `docs.vnrvjiet.ac.in`: extracted, scoped to linked documents

**Date:** 2026-09-30
**Status:** DECIDED — previously BLOCKED

**What changed.** VNRVJIET's IT team reported that the `robots.txt` on
`docs.vnrvjiet.ac.in` (`Disallow: /`, and specifically `/assets/`) is not in
their control and cannot be changed. The project decided that VJLM must extract
everything the official site publishes. CORPUS-001 showed what the block cost:
the rulebooks (B.Tech R25/R22/A19/R19/R18, M.Tech R25/R22/A18/18, MCA R25),
the condonation forms and the scholarship documents exist **only** there, so the
knowledge base could not answer "what is the minimum attendance" at all.

**Scope — narrower than "crawl the host".**
- Only **document** URLs (`.pdf`, `.docx`, `.xlsx`) that the official `www`
  site itself links to, from its bundle or its rendered pages. This is exactly
  what a student gets by pressing "Download PDF". The host is never crawled,
  listed, or walked; nothing is fetched from it that `www` does not link.
- `/nirf2026patent` on `www` — a page `www` links to — is included on the same
  basis. `/api/` (data endpoints, not pages) and `/admin` (CMS login) stay out.
- Same politeness as every other fetch: an honest user agent naming the
  project, rate-limited workers, and content-hash change detection, so an
  unchanged file is never downloaded twice.

**Reversible in one line:** `ROBOTS_OVERRIDE = False` in
`ingestion/crawler/discover.py` restores strict compliance.

**Measured effect (dry run of discovery):** 872 → 2,167 sources, of which 1,285
documents on `docs.*`; 0 existing sources renamed, 0 dropped.

**Obligations that come with it.** `docs.*` holds 34+ further faculty CVs and
list-style documents with student names. Existing policy applies unchanged:
faculty documents are RAG-only, never training data (no deletion path from
weights), and PII is redacted at prompt-build time.

---

## INGEST-004 — Every web page in the knowledge base was replaced by the SPA shell

**Date:** 2026-09-30
**Status:** FIXED in the pipeline; knowledge base rebuilt only by running it

### What happened

All 106 page snapshots became the identical 4,045-byte app shell
(`<!doctype html>…`), and extraction turned each into one chunk of meta tags —
`content="VNR Vignana Jyothi Institute…"` — in place of the anti-ragging cell,
admissions, calendar page, departments. 389 good page chunks became 106 junk
ones. The last step was mine: a full re-extraction (INGEST-003) read snapshots
that were already the shell.

### Root causes — four, compounding

1. **`refresh.py` sent every due id to `fetch.py`**, pages included, and
   `fetch.py` did not filter by type. Pages were fetched over plain HTTP, which
   on this client-rendered site returns the same shell for every route, and the
   shell was written over the rendered snapshot.
2. **Playwright, Chromium, bs4 and lxml were not installed** on this machine,
   so `render.py` could never have repaired it. The pipeline's 106 pages had
   never once been rendered here.
3. **`extract.py` accepted markup as page text.** Nothing checked that a page
   snapshot was `render.py`'s output.
4. **`render.py`'s shrink guard was poisoned by the shell.** It compares a new
   render to the previous snapshot's size; with the 4 KB shell as "previous",
   genuine renders of short pages looked like collapses and were rejected.

### Fixes (all in pipeline code)

| stage | fix |
|---|---|
| `fetch.py` | never fetches pages, whoever calls it; a "document" answering with anything but its file signature (`%PDF`, `PK`) is a failed fetch, not a change |
| `refresh.py` | documents vs pages split by type (docx/xlsx included); a page with no rendered text is **always due**; never-rendered pages render even without a site deploy; `--force` rechecks pages; the CLI waits for the dense build |
| `extract.py` | a page snapshot that is raw markup → `shell`, excluded and reported; a document that fails to parse keeps its last-good chunks, in full runs too |
| `render.py` | shrink guard compares only against a snapshot that was actually rendered |
| environment | `playwright`, Chromium, `beautifulsoup4`, `lxml` installed |

### What rendering was also missing — found while fixing it

- **Hidden content.** The renderer read each page as loaded and clicked
  nothing. React only mounts the active tab: `/academics/regulations` showed
  B.Tech only — M.Tech and MCA existed only behind bare `<button>`s with no ARIA
  role. It now clicks every content control (tabs, accordions, plain buttons),
  skipping header/nav/footer/forms and anything labelled as an action (submit,
  pay, apply, login…), rescans after each click for nested tabs, and returns to
  the page if a button navigates. *A `<button>` without a `type` attribute has
  `type="submit"` by default — excluding "submit" buttons that way silently
  excluded every plain button on the site.*
- **Links only a rendered page contains.** Faculty profiles, blog posts and
  the M.Tech rulebooks are not literal URLs in the bundle. `render.py` now
  records every link it sees; `discover.py` merges them, and new links trigger
  discovery on the next cycle, so the crawl converges without a timer.
- **Discovery gaps:** root-relative `/docs/…` PDFs (24 on `www`, incl. TGEAPCET
  cutoff ranks and CSE placements 2025-26), `.docx`/`.xlsx` (read with the
  standard library), routes declared in the router but absent from the sitemap
  (`/page/examination-cell`, `/rti`, `/ugc-compliance`…), filenames containing
  spaces, and source ids colliding across folders (now suffixed, not dropped).
- **A regression I introduced and caught:** replacing the section-route pattern
  instead of adding to it retired 8 club pages. Restored; the dry run confirms
  0 sources dropped.

Tests: `ingestion/test_refresh.py` 21/21 pass.

### Broken on the site itself — nothing to fetch

`/documents/Transport-Circular-AY-2026-27.pdf` and
`/docs/circulars/ACAD-C141-Elective-OE-I-2627.pdf` return 404. The pipeline
records them as errors and retries each cycle, so they arrive if fixed.

---

## INGEST-005 — Chunks must provably match their snapshot

**Date:** 2026-09-30  **Status:** FIXED

Incremental extraction assumed a source's chunks were always made from its
current snapshot, and re-extracted only sources whose snapshot changed *in this
cycle*. Three pages re-rendered outside a cycle (a smoke test) came back
"unchanged" in the next one and kept chunks made from the old SPA shell —
`/about/anti-ragging-cell`, `/academics/regulations`, `/departments/cse`.

Every chunk now carries `src_hash`, the content hash of the snapshot it came
from. Each incremental run also re-extracts any source whose chunks do not
match its current snapshot, so the knowledge base repairs itself from any
drift, whatever caused it. Chunks from before this field existed count as a
mismatch once, which makes the first run after the change a full re-extract.

**Also:** the site's own 404 page (`404 Oops! Page not found`, 39 characters)
is now recorded as `not_found`, not an error. Routes like `/about-us`, `/rti`
and `/icc` are referenced by the app but do not exist; as "pages without
rendered text" they would otherwise be re-rendered on every cycle forever.

---

## INGEST-006 — Scanned documents arrive progressively, not all at once

**Date:** 2026-09-30  **Status:** DECIDED

**Measured:** a sample of 80 newly fetched `docs.*` PDFs was **60% scans**,
averaging ~7 pages each. Across ~1,400 new documents that is ~5,500 pages of
OCR — at the original ~24 s a page, about **36 hours** in one extract run,
during which no web page or text PDF from the same cycle would be searchable.

**Speed — the orientation classifier.** RapidOCR's text-direction classifier
took 85% of the time: **69 s vs 9.9 s** for the same page under load, with
identical output (147 lines, the key date present). It only helps with lines
printed upside down, which scanned notices do not have; whole-page rotation is
applied by pdfium from the PDF. Disabled.

- **Rejected: 150 DPI.** 7.7 s vs 9.9 s, but 2 of 147 lines lost.
- **Rejected for now: parallel OCR processes.** 4 processes × 5 threads was
  only ~1.4× the throughput of one process (15.3 vs 21.1 s a page, measured
  while rendering competed for CPU). Not worth the process management yet;
  revisit if the backlog proves too slow on an idle machine.

**Structure — a budget and a backlog.** `extract.py` spends at most
`VJLM_OCR_BUDGET_S` (default 1,200 s) on OCR per run. Scans beyond it are
*deferred* to `preprocessing/out/ocr_backlog.json`; the run completes and is
indexed. Every later run targets the backlog first; cached scans are free. The
refresher treats a non-empty backlog as work in progress: it always extracts,
and sleeps 60 s between cycles instead of 15 minutes. A deferred document that
had chunks before keeps them (last-known-good).

Net effect: pages and text documents become searchable in the cycle that
fetched them; scanned documents follow in ~20-minute batches, each one
searchable as soon as its batch is indexed. Nothing is left out; it just
arrives in order of cost.

---

## INGEST-007 — Render speed, and work a stopped cycle must not lose

**Date:** 2026-09-30  **Status:** DECIDED

**Why refresh felt fast before and slow now.** Earlier cycles were fast partly
because they did not do the job: pages were never rendered (plain-HTTP shells,
INGEST-004), 1,252 `docs.*` files were skipped, scans were dropped. The first
full run is a one-time catch-up. But opening every tab (INGEST-004) also made
steady state slower: ~630 pages at ~10 s+ each, so a site deploy meant ~1h45m of
re-rendering. Profiled on `/departments/it`:

| phase | time | fix |
|---|---|---|
| page load, `wait_until="networkidle"` | **19.1 s** of 32 | → `domcontentloaded` + the existing stable-text poll: 1.0–1.8 s vs 1.9–12.4 s on 4 pages, **byte-identical text** on every one. The site never goes network-idle (analytics beacons, chat widget polling). |
| 21 clicks + 350 ms settle | 9.4 s | kept — settle guards against animated reveals |
| re-extracting the page after each click | 3.3 s (worse under load) | a text fingerprint (djb2 of `innerText`, in the browser) — full extraction only when a click revealed something new; output identical on 8 pages |

Pages also render in **4 parallel browsers** (`VJLM_RENDER_WORKERS`), the same
concurrency `fetch.py` already uses on the same server. Measured under heavy
background load: 25 s/page vs 38 s/page serial on the 8 heaviest department
pages. Timings under that load were noisy (one serial rerun: 47 s); throughput
will be measured on an idle run, not claimed from these.

**A stopped cycle lost what it fetched.** `extract.py` only processed what the
current cycle saw change. A document fetched by a cycle that was then stopped
looked "unchanged" to every later cycle and never reached the knowledge base.
`preprocessing/out/extracted.json` now records, per source, the snapshot hash it
was last extracted from; anything whose current snapshot was never extracted is
pending, and `refresh.py` runs extraction while any is pending. Measured before
the restart that needed it: **2,273 sources pending**, including the 1,405 new
documents a stopped cycle had fetched.

## APP-001 — Conversation is not a question

"hi" went to retrieval as a search query, returned five irrelevant passages (a
newsletter matching "hi"), and the fine-tuned model correctly refused — shown to
the user as "No grounded answer in the sources". Greetings, thanks and "what are
you / what can you do" now get a direct reply from `app/server.py`: no
retrieval, no sources, no model call. Anything containing a real question still
goes to retrieval ("hi what is the fee", "help me with fees").

---

## INGEST-008 — The laptop slept, and the pipeline stalled

**Date:** 2026-09-30  **Status:** FIXED (effective from the next app start)

A 457-page render made ~80 pages in 30+ minutes. The log showed
`ERR_NETWORK_IO_SUSPENDED` and `ERR_NETWORK_CHANGED`: Windows idle-sleep
suspends network I/O, so rendering stalled, and pages loading at that moment
came back with 0 characters (recorded as failures, retried next cycle — no
knowledge lost, but the run crawled).

`refresh.py` now calls `SetThreadExecutionState(ES_CONTINUOUS |
ES_SYSTEM_REQUIRED)` when a cycle starts and releases it when the cycle ends —
what a download manager does. Idle sleep is held off only while a cycle is
working; a closed lid or a manual Sleep still sleeps, and no system setting
changes. A no-op on other operating systems.

**Known limitation, not yet fixed:** `render.py` writes its state once, at the
end. Stopping it mid-run (a restart, a crash) discards the state of pages it
already rendered — their snapshot files exist, but the next run renders them
again. Costs time, never knowledge. Fix when it matters: checkpoint state every
N pages, as `index.py` already does for embeddings.

**Update (21:55): the first fix was insufficient.** The event log showed the
laptop entering **Modern Standby** three times during cycles — 18:43→20:47 (the
"two-hour extract hang"), 21:07→21:16 and 21:17→21:52 (the "stuck fetch").
Modern Standby starts when the *screen* turns off, and `ES_SYSTEM_REQUIRED`
alone does not prevent it. `_stay_awake` now also requests
`ES_DISPLAY_REQUIRED` — what a video player asks for — held only while a cycle
works. Nothing in the pipeline had hung; every "hang" today was the machine
asleep. Verified by effect (no standby events during the following cycle), since
`powercfg /requests` needs an elevated prompt.

## RAG-SCOPE-001: scope retrieval by regulation, programme and source family
**Date:** 2026-10-12  **Status:** ACTIVE (rag/scope.py, rag/retrieve.py)

First student test (live app, role=student, mode=coverage): 8/19 = 42% answer
accuracy; retrieval hit@5 31%. Failures were retrieval, not the model: B.Tech R25
questions pulled R18/R19/R22 and M.Tech rulebooks ("O grade = 4", CIE 30+10); 62%
of chunks are syllabus text and ~5% faculty CVs; the R25 table of contents "covers"
every query; coverage re-ranked only BM25's top 60.

Changes, all in coverage's candidate/ranking stage: wide pool (top 300 + 120 per
family); regulation code and programme read from the question and matched to each
rulebook (default current regulation R25, B.Tech over M.Tech/MCA); priors that demote
faculty/report/syllabus unless the question is about them; table-of-contents penalty;
document-identity words (r25, btech, vnrvjiet) not scored at chunk level; plural/singular
matching; section-heading boost; BM25 tie-break; up to 3 chunks per rulebook; family
quotas; student-vocabulary synonyms (clear->pass, subject->course). Every constant in
scope.W is an assumption, tuned on the DEV split only.

**Measured (retrieval hit@5, evaluation/eval_student_retrieval.py, no LLM):**
dev 33% -> 88%, test 29% -> 71%, fresh holdout (written after tuning) 65% -> 100% with
scoping on vs off; all 62 questions 85%; BM25 with the same scoping 73%.
**Caveats:** the test misses were seen before one final change (2 chunks per site page);
the holdout skews to website facts, which the site prior favours, so the rulebook-heavy
dev+test figure (~80%) is the more representative one. Dense retrieval was disabled during
all of this (vectors not yet rebuilt for 98k chunks).

## APP-002: citations computed by code; prompt/context fixes for the fine-tuned model
**Date:** 2026-10-12  **Status:** ACTIVE (rag/answer.py, rag/focus.py, app/server.py, app.js)

Review of two live answers found: truncated rule (75% only; the 65-75% condonation band
and "<65% never condoned" missing), an unsupported hedge ("check against your own
regulation batch, R18/R19/R22 differ"), pasted clause text, a Source line citing the
/payment page for facts only the fee-guidelines PDF contains, and a wrong "[mca]" tag.

Root cause is in training, not the prompt: training/build_dataset.py trains on ONE-passage
contexts with the answer = the passage's first good sentence + a "Source:" line, and the
role-differentiation rows hard-code the "regulation batch" hedge. At inference the model
gets five passages and copies the first sentence of the first. Rewording the system prompt
did not help (the added rules made it paste the first passage and drop the Rupay fact), so
the original in-distribution prompt was kept.

Fixes in code, none of which lets the model say anything not in a retrieved passage:
the model's own Source line and the learned hedge are removed; each answer sentence is
attributed to the passage that contains its words ([n] inline, "Sources used" in the UI,
unused passages dimmed, unsupported sentences flagged); passages are re-ordered question-
first so "copy the first sentence" copies the right one (rag/focus.py); the following
clauses of the same numbered rule and any question-matching sentence from another passage
are appended as "Also stated in the sources". Department tags are no longer inferred from
body text on general pages (preprocessing/enrich.py) - /payment was tagged "mca".
Student role note no longer uses the exact cue that triggered the hedge.

**Not fixed, and the real fix:** retrain with multi-sentence targets and multi-passage
contexts (training/build_dataset.py), without the hard-coded hedge. Until then the model
behaves as a first-sentence extractor and the code-side steps above carry completeness.

## RAG-EVAL-002: coverage stays; identifier rescue; answering instead of refusing
**Date:** 2026-10-12  **Status:** ACTIVE (rag/scope.py, rag/answer.py, rag/synonyms.py, app/server.py)
Retraining is ruled out (compute), and coverage is the required retrieval measure, so
everything below is retrieval- and code-side. Numbers are hit@5 on the 62-question
student set (evaluation/eval_student_retrieval.py, --mode coverage --split all).

**Dense was rebuilt (104,412 x 384, rows == chunks) and measured, then not adopted.**
coverage 84% / hybrid3 74% / hybrid 69% / dense 63% (hit@1 56/55/47/44). Dense alone
rescues 3 questions coverage misses (CIE-SEE marks, repeat-exam paraphrase, NAAC grade)
but loses 16; blending loses 7 for 1. Coverage is both the required measure and the best
one here, so the comparison is recorded and dense stays off.

**Two things assumed and measured false.** (a) Dense was expected to be the biggest lever;
it is the worst single mode. (b) Grade/credit tables were assumed mangled by PDF
extraction and queued for OCR; the corpus holds them clean ("Table 6: Letter Grades and
Grade Points ... >= 90 O Outstanding 10"). Both theories died in ten minutes of
measurement, and no OCR work was done.

**The actual defect: queries lose their most discriminating term.** rag.index.tok
lowercases and drops single characters, so "what grade point is given for the O grade in
B.Tech R25" is searched as [grade, point, given] - the O is gone and every grade table in
every regulation covers those terms equally. Fixed by rescoring the candidate pool against
raw chunk text with case intact (scope.identifiers / id_weights / id_bonus, W["ident"]
0.6), on coverage's own scale beside W["head"]. Rebuilding BM25 with a tokenizer that
keeps single letters would mean re-indexing 104k chunks; this does not.
Two wider variants were measured and rejected, both 84% (no gain): a flat per-identifier
bonus (fixed r04+a01, broke c01+d01 - the roman numeral in "II year", idf 1.65, outweighed
real coverage), and IDF-weighted over all acronyms (fixed r04, broke c01). Restricting it
to terms the index does not contain - exactly what the tokenizer destroyed - gives
**84% -> 85%, hit@1 56%, MRR 0.691 -> 0.697, no regressions.**

**Refusing when the answer was retrieved (APP-003, rag/answer.py: rescue).** The model
answered "I could not find this" with the R25 attendance clause at rank 1, because the
question says `sit the exam` where the rule says `appear for the Semester End
Examination`. When the model refuses, the top passage's own sentences are quoted instead.
The retrieval score cannot gate this - "who won the IPL in 2020" scores 0.788 and the
answerable attendance paraphrase 0.787 - so three structural guards carry the safety:
requests for the reader's own records are never rescued (wh-word + "my"), one of the
question's three rarest terms must be in the passage (via inflections and domain
synonyms), and the quoted sentence must touch the question twice over (one term let a
placements page answer "share price of Infosys"). 9/9 on a probe of 3 answerable and 6
out-of-scope questions, with all six refusals intact.
IDF measures corpus rarity, not topicality: in a rulebook `sit` (7.49) and `exams` (6.69)
are rarer than `attendance` (5.25), so student phrasing looked like the topic. Hence
sit/write/sitting -> appear in synonyms.py DOMAIN; retrieval unchanged at 85% after it.

**Answers with no support at all.** The model sometimes returns only its trained-in hedge
("For reference when advising students: confirm which regulation batch applies ... the
provisions differ across regulation versions") as the ENTIRE answer - it was the whole of
one credits answer. The wording varies each time, so it is caught by what it lacks:
after attribution, an answer no retrieved passage supports is replaced by the passage's
own sentences (app/server.py, both the JSON and the streaming path). The hedge filter
also now keys on "regulation batch"/"regulation version", the training template's words -
the regulations say "R22 Regulations" and are untouched.

**Excluding facts (rag/focus.py).** "which cards don't work" scored too low against
"Rupay Credit & Rupay Debit Cards are not allowed to pay the fee" for the general
supplement to reach it, so a question asking what is excluded now pulls prohibition
sentences at a lower bar - but still through the question's own scorer, because matching
on any shared word dragged a hostel rule about overnight guests into a fees answer.

**Measurement was not repeatable.** Generation ran at temperature 0.2, so re-measuring
shuffled one or two of 26 questions - 3.8% each - and a real change could not be told
from the sampler. VJLM_TEMP / VJLM_SEED now control it (app/server.py) and accuracy runs
use VJLM_TEMP=0.

**End-to-end, role=student mode=coverage, 26 questions (evaluation/student_accuracy_coverage.py):
81% -> 92% overall (24/26), 77% -> 91% answerable (20/22), literal 11/12 -> 12/12,
unanswerable 4/4 throughout.** The two left are a retrieval miss (the no-dues form IS in
the corpus, 3 chunks, listing the sections to clear) and an extraction one (the II-year
calendar extracts as "ACADEMICCALENDAR ... 15th June 2026- | 30th", columns mashed).
26 questions is a small set: each is 3.8%, so this is "about 85-95%", not 92% exactly.

## DATA-003: document dates cannot settle which source is current
**Date:** 2026-10-12  **Status:** INVESTIGATED, not adopted for ranking
The corpus contradicts itself: VNR-Online-Fee-Payment-Website-Guidelines.pdf says "Rupay
Credit & Rupay Debit Cards are not allowed to pay the fee", and
VNR-VJIET-Online-Fee-payment-terms-conditions.pdf lists "RuPay Cards" among accepted
methods. Both public, both authentic.

Three date sources were checked and none settles it:
* **Crawl timestamps** say when WE fetched a file, not when the college published it.
* **HTTP Last-Modified** looked decisive (the guidelines PDF: Thu, 20 Aug 2026) until it
  was checked across the host: ALL 1,661 docs.vnrvjiet.ac.in files return 20 Aug 2026
  08:14:2x, seconds apart, including a 2020 NIRF report. It is a bulk migration
  timestamp. www.vnrvjiet.ac.in sends no Last-Modified at all - which is the host the
  terms-and-conditions PDF sits on, so the two cannot even be compared.
* **PDF /CreationDate** is the only differentiating signal and points the other way:
  guidelines authored 2024-08-06, terms-and-conditions 2026-05-14. It records when the
  Word original was authored, not when the college published it.

So recency is NOT used for ranking. fetch.py does now record Last-Modified (correct, free,
and other hosts may populate it), Retriever exposes it as `published` and the citation
payload carries it, but nothing scores on it. Resolving the contradiction needs a human
decision or the college's own answer; preprocessing/doc_dates.py reads PDF internal dates
if that signal is ever wanted. Both contradicting sentences are shown with their own
citations rather than either being asserted.

Also found: sync_state.json has two concurrent writers. The live refresher wrote its
whole in-memory state over the date backfill's results and erased them. backfill_dates.py
now re-reads and merges before writing; the refresher itself still has the race.


## RAG-EVAL-003: coverage is the default retriever; three fixes on top of it

Coverage (sir's 1 / 0.5 / 0 synonym weighting, scope-adjusted) was already the best
measured mode (RAG-EVAL-002: 84% hit@5 vs hybrid3 74%, hybrid 69%, dense 63%) but the
server, the UI and `Retriever.search` still defaulted to `hybrid`, and fell back to BM25
when the dense index was missing. All three now default to coverage, and the fallback is
coverage. No retraining, no re-indexing.

Failure analysis of the 9 misses (hit@5 on 62 questions) found three causes that were
about the retriever, not the data:

1. Question scaffolding scored as content. IDF measures rarity in a rulebook that says
   "shall", so "does" (5.33), "many" (5.02) and "have" (3.47) came out RARER than "naac"
   (4.25). "What NAAC grade does VNRVJIET have?" ranked a grading-rules chunk (grade +
   does + have = 0.75 coverage) above the accreditation page. `scope.QUESTION_SCAFFOLD`
   (does did many much have has there stand stands mean means) is dropped from the query
   only. A wider list (do, get, give, any, tell ...) broke r11 and r12 and was rejected.
2. A document's identity can live only in its file name. The CSBS course tables never
   say "CSBS"; `II-B.Tech-CSBS-R25.pdf` does, so all branches tied. A query term found in
   the chunk's URL now adds `W["url"]` (0.6) x its IDF share, only to chunks that already
   cover something.
3. Definition questions ("what does SGPA stand for") are answered by the abbreviation
   table, which mentions the term once and so loses to the many chunks that use it. A
   definition-intent query adds the glossary chunks to the candidate pool and gives those
   that contain the identifier `W["glossary"]` (0.6).

Measured, 62 questions, coverage, hit@5 / hit@1 / MRR:
  before   85% / 56% / 0.697
  1 only   87% / 58% / 0.713
  1 + 3    89% / 60% / 0.729
  1 + 2 + 3 90% / 63% / 0.759     (dev 92%, hold 94%, test 86%; no question got worse)
Tried and not kept: extra domain synonyms (carry, repeat, miss, clubs): no change.

End-to-end (26 questions, deterministic) first dropped 24 -> 23: Q7 (tuition fee) kept
its retrieval hit but the model copied the table caption, not the sentence holding the
figure. The passage's own best-matching sentence scored 1.0 against the question.
`focus.supplement` step 1c now appends that sentence when the answer left it out (score
>= 0.9, one sentence). Result: 24/26 = 92% again (answerable 20/22, literal 12/12,
unanswerable 4/4).

Still failing: Q19 (no-dues clearances: the form is in the corpus but the question's rare
word "clearances" is absent from it; a clearance->certificate/dues synonym moves it from
rank 12 to 6, not into the top 5) and Q20 (calendar PDF extraction). Remaining retrieval
misses: r03, r13, l02, k01, d02, h17. The 62-question set was used to find these
causes, so the 90% is a development figure; the e2e set is separate.

---

### RAG-EVAL-004 - three-signal fusion, and what the semantic arm is actually for
2026-10-18

The hybrid is now the three weighted signals the design calls for, not two:
`FUSION = (keyword, WordNet synonym, semantic) = (1.0, 0.5, 1.5)`, all non-zero,
in `rag/retrieve.py` under `mode="fusion"`. `scope.cover_parts()` is what made
this possible: coverage used to return one number for "term or synonym", so
keyword and synonym evidence could not carry different weights. It now returns
the two shares of the query's IDF mass separately.

Two findings, both of which cost real accuracy before they were fixed:

1. **The embeddings were never retrieving anything.** The candidate pool was
   built from the BM25 ranking, so coverage and cosine only ever RE-RANKED what
   keyword matching had already found. `evaluation/diagnose_misses.py` put 14 of
   25 misses at BM25 rank 300-9000 - unreachable at any weighting - while cosine
   ranked those same chunks 2nd, 16th, 22nd, 37th. The reverse also happens
   ("leftover food at the hostels": BM25 7, cosine 908), so neither arm can own
   the pool. The pool is now the UNION of the BM25 ranking and the cosine top
   300 (`VJLM_DENSE_POOL`), and every candidate is scored by all three signals.
2. **The per-document caps were a recall ceiling.** Three chunks of a rulebook
   and one of a syllabus PDF was a presentation rule for a 5-passage prompt, but
   it applied at every depth: a clause ranked 4th within the R25 PDF could not
   be returned at ANY topk. Lifting the caps moved hit@5 79% -> 83% and hit@100
   89% -> 95%. Reserving even ONE slot for a second source cost 2 points, so no
   slot is reserved.

Measured on 98 questions (62 tuned + 36 written from the corpus and never tuned
on), `evaluation/depth_curve.py`:

              @1    @3    @5    @10   @20   @50   @100
  coverage    -     -     74%   79%   84%   87%   88%
  fusion      58%   77%   83%   88%   92%   95%   95%

**Six attempts to convert hit@20 into hit@5 were all rejected by measurement**,
which is why the gap is still there and why it should not be attacked again
without a new idea:

  - cross-encoder rerank (ms-marco-MiniLM-L-6-v2): 83% -> 73%, 664 ms/query
  - best-sentence rerank of the leaders: 81% at weight 1.0, 79% at 2.0
  - bonus for section-overview pages, derived from the URL tree: 83%, unchanged
  - IDF raised to a power, to favour the rare term: flat 83% at 1.5, 2.0, 3.0
  - extra domain synonyms: no change
  - larger dense pool (300 vs 100) in the old BM25-only design: no gain

`evaluation/why_miss.py` says why, per arm. In 6 of 12 reachable misses the
chunk that wins covers MORE of the question than the chunk holding the answer -
often all of it. "In which area of Hyderabad is the campus?" is answered on
/about, which covers 0.64 of the question, and lost to /campus-life, which
covers 1.00, because the answer is a place name the question cannot contain.
No reweighting of the question's terms can fix that; the winning page really is
a better match for the words asked.

**Embedding model: `BAAI/bge-small-en-v1.5` stays.** `evaluation/screen_embedders.py`
re-scored the pools with each candidate: bge-small 77%, arctic-embed-xs 76%,
and the larger builds cannot be fetched on this machine (Windows symlink
privilege). A BGE query instruction prefix (`VJLM_QPREFIX`) is a small, free
gain and is on. Semantic scores are min-maxed over the CANDIDATES, not a
corpus-wide percentile, so all three arms span 0-1 and the weights compare.

**The prompt's character budget is now fixed** (`PROMPT_CHARS`, 3250 = the old
5 x 650) and the passage count is `VJLM_TOPK`. Carrying more passages therefore
costs no context, no latency and no memory - each is trimmed harder by
`rag/focus.py`, which keeps the sentences matching the question. Since the
answer is in the top 10 for 88% of questions and the top 20 for 92%, showing
the model more of what retrieval already found is the cheap lever that the six
reranking attempts were not.

**Retrieval depth and prompt diversity are two different jobs** (2026-10-18).
Lifting the per-document caps raised hit@5 79% -> 83%, and the end-to-end test
then fell 24/26 -> 22/26 for a reason hit@5 cannot see: with no cap, ONE
document monopolises the prompt. "Where can I study quietly and borrow books"
returned four passages of `Adminstrative Manual.pdf` and never showed the
library page; "how many credits is Design and Analysis of Algorithms in CSBS"
showed four different branches' syllabus tables and the model answered from the
wrong branch. The cap is now a SHARE of topk (`DOC_CAP_FRAC`, 0.4) with a floor
of 2, so it binds on a 5-passage prompt and vanishes at evaluation depth. That
costs retrieval nothing - hit@5/@10/@20/@100 are 83/89/94/98% at cap 1.0, 0.6
and 0.4 alike - so the damage was never caused by capping, but by the FIXED
per-family values it used (one chunk per syllabus PDF, three per rulebook).

**Reciprocal-rank fusion: rejected.** Combining the three arms by rank instead
of by weighted score is scale-free and is the textbook way to mix a hybrid, so
it was worth one measurement: hit@5 83% -> 72%, hit@1 58% -> 44% (`VJLM_RRF`,
kept at 0). The arms' magnitudes carry information that ranks discard - how
MUCH of the query a chunk covers is the signal, not merely that it covers more
than the next chunk. That is the seventh ordering idea rejected by measurement.

**Branch mismatch is now penalised like regulation mismatch** (2026-10-18).
`scope.parse_query` knew the programme (B.Tech / M.Tech / MCA) but not the
BRANCH, so the four near-identical syllabus PDFs of CSBS, IOT, CYS and AIDS were
separated only by a weak file-name term bonus. "How many credits is Design and
Analysis of Algorithms in CSBS" retrieved the right document AND answered from
the wrong branch, which is a failure hit@5 scores as a success. A question that
names a branch now boosts that branch's documents (`branch_match` 1.5) and
penalises another branch's (`branch_mismatch` 0.35), leaving documents of no
particular branch untouched - the same shape as the regulation priors in
RAG-SCOPE-001. The branch comes from the enriched `department` field where it
exists (38k chunks) and from the file name otherwise, which is the only place a
syllabus PDF names its branch. Retrieval is unchanged by it (hit@5 83%, hit@1
58% -> 60%), which is expected: what it fixes is what reaches the prompt.

**Enumeration bonus: rejected.** An enumeration page answers a "which X" question
by repeating X - /campus-life/clubs says "club" a dozen times while listing them
and shares nothing else with "which societies and clubs can I join" - so it was
ranked below forty individual club pages. Coverage measures breadth and counts
each term once, by design, so an IDF-weighted repetition bonus was added behind
`W["enum"]` for list-intent questions: hit@5 83% at weight 0.5 (unchanged) and
81% at 1.0. Kept at 0. Eighth ordering idea rejected by measurement.

**The two questions retrieval cannot reach at all, diagnosed** (deep pool, 98
questions, hit@100 98%): `k01` is an ORDERING failure, not recall - the clubs
index sits at cosine rank 511, inside the 1200 pool, and simply scores below the
individual club pages. `n31` ("which bodies has the college been approved by")
is a CHUNKING failure: the whole /about page is one chunk of landing-page copy
with "approved by AICTE" buried in it, at cosine rank 24643. No weighting
reaches that; splitting the page by its headings is the only fix, which is what
the structural chunker is for.

**The calendars' vocabulary bridge** (2026-10-18). Both failing calendar
questions - the end-to-end Q20 and d02 - had the right page extracted, the OCR
readable and the dates present ("06th July 2026"), and still missed: the student
writes "when do classes START for SECOND year", the calendar says "Commencement
of Class Work" for "B.Tech. II Year". start/begin -> commencement and
second/third/fourth -> ii/iii/iv were added to the domain map. `commencement`
has IDF 5.91 in this corpus, so it discriminates strongly once a query can reach
it. Measured: hit@10 89% -> 90%, hit@5 unchanged. The calendar PDF itself is
still outranked by the /academics pages, which cover more of the question's
words - the same shape as every remaining miss - but the class-commencement
circular now surfaces inside the top 10, which is what a 10-passage prompt sees.

**More passages do NOT convert into more accuracy** (2026-10-18). This was the
one lever the eight rejected reranking ideas left: retrieval puts the answer in
the top 5 for 83% of questions but the top 10 for 90%, the prompt's character
budget is fixed, so ten passages cost no context - each is simply trimmed harder
by rag/focus.py. Measured end-to-end, 26 questions, fusion, deterministic:

  5 passages (650 chars each)   23/26 = 88%   answerable 19/22   median 22 s
  10 passages (325 chars each)  22/26 = 85%   answerable 18/22   median 30 s
  20 passages (220 chars each)  17/26 = 65%   answerable 15/22

The retrieval gain does not reach the answer. One question flipped and it flipped
the wrong way: "How is my CGPA worked out?" was answered "My Big Idea Competition
- annual flagship competition that invites students to pitch breakthrough
ideas", which is a passage the 10-deep prompt added and the model preferred.
That is distraction, and it is the known behaviour of a small instruction-tuned
model given many short snippets. Latency also rose 36% on prefill.

So topk stays 5. The conclusion worth keeping is where the bottleneck now is: it
is no longer retrieval. On the 22 answerable questions the model gets 19 right
from 5 passages, and the failures are not "the answer was not retrieved" but
"the answer was retrieved and not used" - Q22 had the library page at slot 3 and
was answered from the Administrative Manual at slots 1-2.

**Ordering the chosen passages by their best sentence: rejected** (2026-10-18).
Reranking the POOL by best sentence was already rejected on retrieval (hit@5 83%
-> 81%), but ordering the five passages retrieval had ALREADY chosen is a
different question that hit@5 cannot answer at all - the set is identical, only
the reading order changes - and this model answers from the early passages. It
was built behind `VJLM_PROMPT_ORDER=sent` and measured end-to-end: 23/26 both
ways, with the SAME three questions failing (19, 20, 22). Even Q22, the question
it was built for, did not move: the Administrative Manual passages that
outranked the library page also score well sentence-wise. Ninth idea rejected.

`DOC_CAP_FRAC` default is 1.0 - see the correction below.

### RAG-EVAL-005 - what the broad question set says, and why 88% was optimistic
2026-10-18

The 26-question student test reports 23/26 = 88% (fusion, 5 passages). Running
the SAME end-to-end path over all 98 corpus-verified questions
(`evaluation/answer_accuracy.py`) gives 56/98 = 57% by strict pattern match. That
gap is partly a measurement artifact and partly real, and both halves matter.

**The measurement artifact.** The 98 questions' gold patterns were written to
match CHUNK TEXT, for scoring retrieval, and they do not match natural answers.
Hand-grading all 42 failures: 7 are correct answers scored wrong - "O
(Outstanding) | 10" against `Outstanding\s*10`, "Controller of Examinations
(COE)" against `CoE\s+Controller of Examinations`, "supplementary semester end
examination" against `Supplementary Examinations`, "established in 2006" against
`Established 2006`. A further 4 are defensible but unprovable. So the real
figure is about 63/98 = 64%, up to 68% if the ambiguous ones count.

**The real gap.** 31 answers are genuinely wrong, in three classes:
  - 17 are retrieval misses - the answer was never in the prompt. These are the
    same misses hit@5 reports (83%), and the passages the model answered from
    instead are plainly off-topic: "what is the longest I can take to finish
    B.Tech" was answered from a student testimonial, "when was the society that
    runs the college started" from the VLSI Society of India.
  - 8 are extraction failures: the right document reached the prompt and the
    model took the wrong field out of a table - CIE/SEE marks, the M.Tech annual
    fee, the total sanctioned intake, the name of a centre's head.
  - 8 are abstentions where the clause exists in the corpus but ranked 6th-36th,
    so the model correctly refused on what it was shown (n04 35% SEE at rank 8,
    n14 supplementary exams at 36).

Two honest consequences. First, **the 88% is a tuned figure**: those 26 questions
and their answer-shaped patterns were used while developing the retriever, and
the v2 questions - written from the corpus afterwards and never tuned on - score
39% strict. Second, **answer accuracy is no longer limited by retrieval alone**:
the model converts roughly 77% of its retrieval hits into correct answers, and
the 8 table-extraction failures are generation, not retrieval. A 90% claim needs
answer-shaped gold on all 98 questions before it can even be measured.

### CORRECTION to the per-document cap (2026-10-18, same day)

The claim above that a share-based per-document cap "costs retrieval nothing -
hit@5/@10/@20/@100 are 83/89/94/98% at cap 1.0, 0.6 and 0.4 alike" was measured
wrongly, and the conclusion drawn from it was wrong.

`evaluation/depth_curve.py` retrieves ONCE at depth 100 and slices. The caps
scale with the request (`grow = topk // 5`), so at depth 100 they do not bind at
all - which is why every cap setting measured identical. The cap only bites on
the 5-passage prompt the app actually builds, and that case was never measured
on the broad set. `evaluation/hit_at_prompt.py` now asks for topk passages the
way the app does:

  topk=5, 98 questions      flat cap 0.4   77/98 = 79%
                            flat cap 0.6   77/98 = 79%
                            family-aware   79/98 = 81%
                            no cap         81/98 = 83%

The cap was costing four questions (r03, p02, n02, n17). n17's answer is at
overall rank 5 and every passage above it is another clause of the same
rulebook, so a cap of two per document replaced clauses 3-5 with worse
documents, and the model then abstained on a question whose answer retrieval had
found. A rulebook question is answered BY the rulebook, several clauses of it.

The monopolisation this cap was introduced to prevent does not actually need it:
with no per-document cap, "where can I study quietly and borrow books" still
shows two library passages, because the per-FAMILY quota (`scope.quota`) already
limits one family to three slots. That quota was the working diversity mechanism
all along. `DOC_CAP_FRAC` is therefore 1.0, with the family-aware exemption kept
for anyone who lowers it.

The general lesson, which cost most of a day: measure the configuration the
system actually serves. A recall curve and a prompt are different questions, and
six cap measurements agreed with each other only because none of them exercised
the cap.
