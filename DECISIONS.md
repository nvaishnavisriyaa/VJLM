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
  | C | tuned | **failed** | — | — | — | — | — | — |

  All *completed* rows are n=147 requested, 147 ok, 0 failed. `grounded_f1` CIs are bootstrap 95% over the 87 answerable items. **Arm C did not train to completion — see `FT-005`.** It is left as `failed`, not quietly dropped, because the A↔C comparison this entry is built around is currently *unmeasured*, and an entry that silently listed four arms would read as though five had been planned.

- **`citation_rate` has a structural ceiling of 0.5918, and must never be read as if 1.0 were achievable.** Its denominator is all 147 items, but only 87 are answerable and an abstention cannot cite; 87/147 = 0.5918. Both tuned arms sit at 0.5102 = **75 of 87 answerable items cited, 86% of what is reachable.** The two arms cite 75 items each and agree on 74 of them, which is convergence on the same behaviour, not a copied number. Mistral *base* also attached citations to **10 items it abstained on**; both tuned arms do that zero times.

- **Established before any tuning, from the base rows alone:** Mistral-7B base scores **abstain_recall 1.000** — it refuses all 60 out-of-scope questions unprompted. **That metric is saturated and cannot improve.** The headroom this fine-tune can actually address is the **14 false abstentions** (refusing when the answer *was* in context) and the **36% citation rate**. Any writeup that reports a strong abstention number as evidence the fine-tune worked would be reporting a property of the base model.
- **What four rows already show — read with the A↔C caveat, since arm C is not in yet:**

  1. **The two base models are not remotely comparable, and the tuned ones nearly are.** Llama-3.2-3B base is *far* worse than Mistral-7B base on every axis: grounded F1 0.2369 vs 0.4401, citation 0.0204 vs 0.3605, and critically `abstain_recall` **0.283 vs 1.000** — it answers 43 of the 60 questions it should refuse. After training, Llama-3B LoRA reaches **0.7567 (0.7166–0.7959)** against Mistral-7B QLoRA's **0.7654 (0.7192–0.8073)**. Those intervals overlap across almost their entire width; on this evaluation the 3B model is **not measurably worse than the 7B one.**
  2. **The cost difference is not marginal.** Arm A trained in **74 minutes** against arm B's **230.6**, and generates at **13.72 s/answer** against **20.16** — 3.1× cheaper to train and 1.5× faster to score, for a difference in grounded F1 of 0.0087, which is well inside the ~0.03 noise floor this entry pre-registered. If that survives arm C and Phase 5, the 7B model is paying 3× the training cost for nothing measurable.
  3. **Fine-tuning mattered more than base-model choice.** The gap between the two *bases* is 0.2032 F1; the gap between the two *tuned* arms is 0.0087. Both arms converge on `abstain_recall` 1.000, `false_abstain` 0 and the same 75 cited items. The behaviour being taught is apparently reachable from either starting point — which is exactly the claim the three-way separation rests on, and the first direct evidence for it.

  **What none of this licenses yet:** a selection. These are templated items auto-derived from the corpus (see the scope limit below), and **arm C failed to train**, so LoRA-vs-QLoRA remains unmeasured. The 3B-matches-7B result in (1) stands on its own — it does not depend on arm C — but it is a *(model, method)* result, not a statement about 3B models in general.

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
  3. **`--resume`**, resuming from the newest `checkpoint-*` in the adapter directory, with every matrix stage idempotent — a training stage is complete when `run.json` exists, a scoring stage when its `(arm, role)` row is already in `generation_results.jsonl`. Re-running the driver after any interruption continues rather than restarting.
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
- **Known cost of detaching:** the run is invisible to the launching session's task tracking, so no completion notification arrives automatically. Progress is exposed through `experiments/matrix.log` and `experiments/matrix_status.json` instead. Note that `matrix_status.json`'s `heartbeat` is written **only at stage boundaries** and is therefore not a liveness signal during a 3.5 h stage — use the log file's modification time.
- **Revisit trigger:** A training stage drops below ~20 min (a much smaller model, or a much faster card), at which point the original "restarting is cheaper than the code" argument becomes true again. Or the run moves to a machine where a scheduler exists and should own it.

## FT-005 — Why arm C (Llama-3.2-3B QLoRA) does not fit in 16 GB

- **Status:** **open — still blocking the A↔C comparison.** Two fix attempts have failed. Progress and the two dead ends are recorded at the bottom; do not read the `--no-fp32-head` section as a resolution.
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
- **Remaining cost, stated:** arm C's recipe now differs from arm B's, which kept the fp32 head harmlessly on a 32k vocab. `C↔B` was already the loose comparison — different family, tokenizer, vocabulary and pretraining — so this adds one more item to a list that already forbids strong claims. Trading precision in the loose comparison to gain it in the tight one is the right direction, but it is a trade, not a free fix.
- **Revisit trigger:** a card with more than 16 GB, or a decision from the above list. Whichever comes first.
- **What survives regardless:** arm C's failure does not touch arms A and B. The finding that tuned Llama-3.2-3B matches tuned Mistral-7B on this evaluation rests on `822c92ed` and `baffeb06`, both complete and both scored on all 147 items.
