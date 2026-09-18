# VJLM — Handoff to a GPU Machine

**Read this first.** Written 2026-09-17, moving off a CPU-only laptop (i7-12700H, 15.7 GB RAM, Intel Iris Xe — no CUDA) onto a machine with a GPU.

> **Update, later the same day — the move has happened.** This document was
> written *before* the transfer and several of its instructions are now
> obsolete. The new machine is an **Intel Core Ultra 7 265 (20 cores), 31.5 GB
> RAM, NVIDIA RTX 2000 Ada, 16 GB VRAM**.
>
> Three things this changes:
>
> 1. **§4's "use fp16 instead of bf16" does not apply here.** That is a T4
>    workaround — Turing has no native bf16 and emulates it. Ada is SM 8.9 with
>    native bf16, so `bf16=True` in the notebooks is *correct* on this card and
>    should stay. `training/train_local.py` detects this rather than assuming.
> 2. **Colab is no longer needed for training.** `FOUND-003` chose it because
>    no local CUDA GPU existed; one does now.
> 3. ~~**`docs/hardware_profile.md` describes the old laptop**~~ — **done
>    2026-09-18.** It has been re-profiled against this machine, with the
>    superseded record kept at the bottom for traceability. The 6–8 GB RAM
>    budget does not bind here; disk (43.6 GB free) and VRAM *logits* do.
>
> The status table in §2 has been updated in place. Sections 3–7 are left as
> written, since they record findings that are still true.

> The **master plan** this project implements is [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) (original: [`docs/VJLM_End_to_End_Implementation_Decision_Plan.docx`](docs/VJLM_End_to_End_Implementation_Decision_Plan.docx)). It defines the 15 phases, the gates between them, and the decision-log discipline. The architecture and data-source figures are in [`docs/figures/`](docs/figures/). Read it before changing approach — this handoff only reports progress *against* it.

---

## 1. What VJLM is

**Vignana Jyothi Language Model** — a local, role-aware AI assistant for VNRVJIET that answers student and faculty questions from the college's own public documents, with citations, running entirely on CPU with no external LLM API at runtime.

Three responsibilities stay separate:

| | Holds | Why |
|---|---|---|
| **Fine-tuning** | How VJLM *behaves* — grounded answering, abstention, role framing, VNR terminology, sensitive-topic tone | Behaviour is stable |
| **RAG** | Everything factual — regulations, syllabi, calendars, fees, placements, faculty | Facts change; a re-index is 5 minutes, a retrain is hours |
| **RBAC** | What the authenticated user may retrieve | Filtering happens **before** retrieval reaches the prompt. **The model is never the security boundary.** |

The project's central claim is **evidence-based engineering**: no component is chosen because it is popular. Every choice is a measured comparison recorded in [`DECISIONS.md`](DECISIONS.md) with rejected options and a revisit trigger. Preserve this — it is what the project is defended on.

---

## 2. Current status — honest

| Layer (per the architecture diagram) | Status | Evidence |
|---|---|---|
| **1. Data — discover / fetch / change-detect** | ✅ Working | 776 of 777 sources fetched. The one failure is the site's own broken link (`/departments/vlsi` renders "Department not found") and the anomaly guard refused to ingest it |
| **1. Data — website pages** | ✅ **Now working** | The site is a client-rendered SPA, so plain HTTP returned an empty 4 KB shell for all 106 routes and the corpus used to be 100% PDF. `ingestion/crawler/render.py` renders them with local headless Chromium: 105 pages, 99 distinct, 462 chunks |
| **1. Data — extract / chunk** | ✅ Working | 13,051 chunks (12,589 PDF + 462 page-derived), 0 parse failures. Extraction is now incremental — one changed document re-reads one document |
| **1. Data — metadata / PII / dedup** | ✅ Working | department on 6,760+ chunks, academic_year on 7,476, PII flagged, 1,660 exact duplicates filtered at retrieval |
| **2b. Knowledge base — BM25** | ✅ Working | Rebuilds in **0.8 s**. Recall@5 0.911, MRR 0.803, p50 4.0 ms on the auto-derived benchmark |
| **2b. Knowledge base — dense/hybrid** | ✅ **Now built** | 13,051 × 384. **22 chunks/sec** with `threads=cpu_count` (the 3/sec figure was the unconfigured default), ~10 min for a full build, resumable. Vectors cached by text hash, so a refresh embeds only what changed |
| **2a. Fine-tuning dataset** | ✅ Working | 1,199 / 147 / 147. Split is by document group *before* augmentation — the previous split shared 32% of its test answers with train, making any fine-tune result unmeasurable |
| **2a. Fine-tune** | ✅ **Two arms trained and scored** | Mistral-7B QLoRA **0.4401 → 0.7654**; Llama-3.2-3B LoRA **0.2369 → 0.7567**. Both reach abstain_recall 1.000 and false_abstain 0. The 3B arm trains 3.1× faster for no measurable loss — see §2.1 |
| **2a. Quantization (ours)** | ❌ Blocked on the fine-tune matrix completing | Script is written and wired |
| **3. Inference — RBAC + prompt + citations** | ✅ Working | Filters before ranking, role boosts, citations rendered |
| **3. Inference — authentication** | ⚠️ Role simulation only | The RBAC gate it feeds is real, but in Phase 1 every role sees the same public data, so it filters nothing in practice |
| **4. Local infra — CPU serving** | ✅ Working | 19.4 tok/s, full answer in **8.75 s** |
| **Scheduler (self-maintaining)** | ✅ **Now wired** | Not Task Scheduler — `ingestion/refresh.py` runs when the app starts. Per-source intervals; a cycle with nothing due costs **0.08 s** |
| **Benchmark-driven model comparison** | ⚠️ **4 of 5 scored** | Arm C blocked on VRAM (`FT-005`), so LoRA-vs-QLoRA is unmeasured. Driver is `training/run_matrix.py`; plan in [`experiments/PLAN.md`](experiments/PLAN.md) |
| **Phase 5 human benchmark** | ❌ **Not started — now the blocker** | `evaluation/questions.jsonl` and `gold_sources.jsonl` are empty files |

**In one line: the data pipeline is complete and automatic, RAG has both
retrievers, fine-tuning is now measured rather than pending — and the
remaining gaps are quantization and, above all, a benchmark honest enough to
judge any of it.**

### 2.1 Phase 8 — the fine-tuning matrix, in progress

Five unique configurations, driven by `training/run_matrix.py`. The untouched
Llama-3.2-3B base is scored **once** and reused as the baseline for both A and
C rather than being run twice as if it were two experiments.

| Arm | Model | Method | State |
|---|---|---|---|
| **B** | Mistral-7B-Instruct-v0.3 | QLoRA r=16 | ✅ trained + scored — `baffeb06` |
| **A** | Llama-3.2-3B-Instruct | LoRA r=16 (bf16 base) | ✅ trained + scored — `b48d1ca6` |
| **AC** | Llama-3.2-3B-Instruct | *none* — shared baseline | ✅ scored once — `822c92ed` |
| **C** | Llama-3.2-3B-Instruct | QLoRA r=16 | ❌ **failed — exceeds 16 GB VRAM**, see `FT-005` |

What each comparison is allowed to claim:

- **A ↔ C** isolates the *method*. Everything else is locked: same examples,
  order, seed, epochs, lr, scheduler, warmup, rank, alpha, dropout, target
  modules, batch, sequence length and chat template. Only the base-weight
  representation changes.
- **C ↔ B** compares Llama-3.2-3B and Mistral-7B under the same QLoRA recipe.
  It does **not** isolate parameter count alone, because model family,
  architecture, tokenizer, vocabulary and pretraining also differ.

Scoring is held at NF4 for **every** arm, including the bf16-trained arm A, so
that inference quantization is not a hidden second variable. Reasoning in
`FT-003`.

**Result so far — the 3B arm is holding its own.** Llama-3.2-3B LoRA scores
0.7567 grounded F1 (95% CI 0.717–0.796) against Mistral-7B QLoRA's 0.7654
(0.719–0.807). The intervals overlap almost entirely, and the 0.0087 gap is
well inside this evaluation's ~0.03 noise floor. Arm A trained in 74 minutes
against arm B's 231, and answers in 13.72 s against 20.16. Full numbers and
caveats in `SLM-001`.

**Confirmed prediction:** arm A peaked at **10.61 GB** VRAM and arm B at
**7.18 GB** — the 3B model used 48% *more* than the 7B one, because
Llama-3.2-3B's 128,256-token vocabulary dominates the cross-entropy tensor.
`docs/hardware_profile.md` called this before either arm ran.

**Known open item — which checkpoint is scored.** Arm B's validation loss
bottoms at step 75 (exactly one epoch, 0.2349) and then rises monotonically
through the whole second epoch to 0.2754, while train loss falls to 0.1443.
The scored adapter is the **final** step-150 one, which is the over-fit side
of that curve — and it still beat the base decisively. `checkpoint-75` is on
disk, so "does one epoch beat two?" can be settled by one eval pass with no
retraining. Not yet decided.

### Why the retrieval numbers do not settle RAG-001

The three-way comparison finally ran (`evaluation/eval_retrieval.py --k 5`, n=90):

| mode | recall@5 | MRR | p50 |
|---|---|---|---|
| bm25 | **0.911** | **0.803** | 4.0 ms |
| dense | 0.667 | 0.509 | 4.0 ms |
| hybrid | 0.789 | 0.722 | 8.8 ms |

BM25 wins — but read §3.2 before concluding anything. These questions are
auto-derived from the passages, so they share vocabulary with their source and
are exactly the case BM25 is best at. Dense retrieval is being scored on a test
built around its weakness. This is evidence that **Phase 5 is the blocking
piece**, not evidence that dense is worse.

---

## 3. Two findings worth carrying forward

### 3.1 A 39× local inference regression — already fixed

`pip install llama-cpp-python` gave a generic, unvectorised build:

| Build | tok/s |
|---|---|
| `llama-cpp-python` CPU wheel | **0.5** |
| Official llama.cpp release (auto-selects `ggml-cpu-alderlake`) | **19.4** |

More threads made the pip build *worse* (0.5 → 0.1 at 14 threads) — the signature of no AVX2. **Never benchmark a model on the pip wheel.** Use the official release binaries; `runtime/` is gitignored and re-downloaded by setup.

### 3.2 The 94.4% retrieval number is real but flattering

The benchmark questions in `evaluation/retrieval_benchmark.jsonl` are auto-derived from passages, so they share vocabulary with their source and favour BM25.

**Demonstrated concretely:** a real student question — *"What is the attendance requirement?"* — retrieved the right R25 regulation PDFs but the wrong pages, and the model correctly abstained instead of inventing a percentage. The document says *"75% of the aggregate"*; the question says *"attendance requirement"*; almost no lexical overlap.

So: **94.4% on the proxy benchmark, and a real question still fails.** That gap is the strongest argument for dense retrieval, and it belongs in the report as a measured finding. Human-written questions are still required before claiming anything about real answer quality.

---

## 4. The training bug the GPU machine must fix first

Training reached the loop on a Colab T4 and reported `6/174 [03:18 < 2:19:11, 0.02 it/s]` — about **2 h 19 m**, too long for a free session.

**Suspected cause: `bf16=True` on a T4.** T4 is Turing (SM 7.5) and has **no native bf16**, so it emulates, which is very slow. On a T4 use `fp16=True`. Only use bf16 on Ampere or newer (A100, L4, RTX 30/40 series).

On the new machine, before a long run:

```python
import torch
torch.cuda.get_device_name(0)
torch.cuda.is_bf16_supported()   # True -> bf16 ok, False -> use fp16
```

Also worth cutting for the first successful run: `max_length` 1024 → 512, `num_train_epochs` 2 → 1. Get *a* trained model end to end, measure it, then scale up.

---

## 5. What to run on the GPU machine

```bash
git clone https://github.com/nvaishnavisriyaa/VJLM.git && cd VJLM
pip install pypdf fastembed flask numpy
```

### 5.1 Rebuild the knowledge base (~15 min, no GPU)

```bash
python ingestion/crawler/discover.py     # source catalog from the live site
python ingestion/crawler/fetch.py        # download + content-hash change detection
python preprocessing/extract.py          # text -> 12k chunks
python preprocessing/enrich.py           # department / year / PII / dedup
python rag/index.py                      # BM25 + dense (dense is the slow part)
```

**Dense embeddings are the first win on the new machine.** They were never built — bge-small on GPU takes seconds instead of 68 minutes. Then run the comparison that has been blocked all along:

```bash
python evaluation/eval_retrieval.py --k 5     # BM25 vs dense vs hybrid, same questions
```

### 5.2 Fine-tune

Either run `training/notebooks/colab_compact.py` (self-contained — scrapes, builds the dataset, trains, quantizes; no uploads) or use the local `training/data/*.jsonl` directly. Apply the fp16 fix from §4 first.

### 5.3 Measure, don't assert

```bash
# start llama-server on the BASE model
python evaluation/eval_generation.py --label base
# restart llama-server on the fine-tuned build
python evaluation/eval_generation.py --label finetuned
python evaluation/eval_generation.py --compare
```

Scores grounded F1, abstention recall/precision, false-abstention count, citation rate. **Record base scores before training** — without them "the fine-tune helped" is unfalsifiable, and that is the one claim the project exists to prove.

### 5.4 Serve

```bash
runtime/start_llama.cmd          # or llama-server -m <gguf> -c 4096 --port 8080
python app/server.py             # http://127.0.0.1:7860
```

`app/server.py` prefers any `quantization/builds/vjlm*.gguf` over the base model, so a fine-tuned build is picked up automatically once it lands.

---

## 6. Open decisions the GPU unlocks

Full matrix in [`experiments/PLAN.md`](experiments/PLAN.md). The headline:

| Candidate | Q4_K_M size | Fits 16 GB CPU? | Train method |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | ~1.0 GB | easily | LoRA (the control, chosen for speed, **not** on merit) |
| Llama-3.2-3B-Instruct | ~2.0 GB | comfortably | LoRA |
| Mistral-7B-Instruct-v0.3 | ~4.4 GB | yes, with margin | **QLoRA** on a 16 GB GPU |
| Llama-3.1-8B-Instruct | ~4.9 GB | tightest | **QLoRA** |

**LoRA vs QLoRA is not a free choice — it falls out of model size.** ≤3B fits in bf16/fp16 on a 15 GB card; 7B/8B does not once optimizer state is counted. So the honest experiment is *(model, method)* pairs. With a bigger GPU, LoRA at 7B becomes possible and that comparison gets cleaner.

`configs/models.yaml` is deliberately all `null`. Do not fill it from a leaderboard — fill it from a measurement.

---

## 7. The blocker only the college can clear

`https://docs.vnrvjiet.ac.in/robots.txt` is `User-agent: * / Disallow: /`, and that subdomain holds **1,252 of the 1,923 PDFs** — including the syllabus books and full regulation rulebooks. On `www` we found only 1 syllabus PDF.

We honour it: the crawler skips that host, and `discover.py` reports `skipped 1252 (robots.txt disallowed)`.

**Ask VNRVJIET IT for written permission**, ideally an `Allow` rule for the crawler's user-agent so no exception logic is needed. Recorded as `INGEST-002` (status: BLOCKED) with all options. Everything else is built on the 671 `www` PDFs, so this is additive — nothing is wasted while permission is sought.

---

## 8. Repo map

```
DECISIONS.md              every choice + evidence + rejected options   <- the project's spine
HANDOFF.md                this file
data_catalog/
  classification.md       fine-tune vs RAG for all 14 website categories
  catalog.csv             777 sources, generated
ingestion/crawler/
  discover.py             sitemap + app-bundle route/asset discovery
  fetch.py                snapshot + hash + new/changed/deleted reconciliation
preprocessing/
  extract.py              pypdf -> normalized chunks with page provenance
  enrich.py               department / academic_year / PII / dedup
rag/
  index.py                BM25 + optional dense (degrades, never breaks)
  retrieve.py             RBAC filter BEFORE ranking, role boosts, prompt builder
evaluation/
  eval_retrieval.py       BM25 vs dense vs hybrid: Recall@k, MRR, latency
  eval_generation.py      base vs fine-tuned: grounded F1, abstention, citations
training/
  build_dataset.py        behavioural dataset (5 parts, no facts)
  notebooks/colab_compact.py   self-contained Colab: scrape -> train -> GGUF
experiments/
  PLAN.md                 what gets compared against what
  *.jsonl                 raw measurements
app/server.py             Flask UI, role selector, citations, local llama-server
docs/
  hardware_profile.md     the old CPU box — replace with the new machine's
  phase1_site_recon.md    9 findings about the VNRVJIET site
```

---

## 9. Things that will bite you

1. **The site is a client-rendered SPA.** Plain HTTP returns an identical 4 KB shell for every route. HTML pages need a headless browser; only the PDFs are fetchable directly. The 12,650 chunks are all PDF-derived.
2. **No `ETag`, no `Last-Modified`** anywhere on the site. The normalized-content hash is the *only* change fingerprint. `configs/ingestion.yaml` pins this — do not switch it to raw HTML, or every CSS redesign triggers a full re-embed.
3. **`<link rel="canonical">` is hardcoded to the homepage on every route.** Use the request URL as the document key, never the HTML canonical, or every page collapses into one document.
4. **The sitemap lists 83 nav pages and zero documents**, and misses 23 live routes including two whole departments (CSBS, RAI). Scoped recursion is mandatory.
5. **Faculty CVs are 487 of the 671 accessible PDFs.** They are RAG-only and must never enter training — you cannot delete a fact from weights, and a distributed GGUF keeps personal data forever.
6. **Python buffers stdout when piped.** Background jobs looked silent for an hour. Use `python -u`.
7. **Colab recycles runtimes and wipes `/content`.** `colab_compact.py` re-scrapes from the live site instead of relying on uploads — this is why.
