# VJLM Experiment Plan — what gets compared, and against what

No component is selected without beating alternatives on the same benchmark, same hardware. Every run appends raw numbers to `experiments/*.jsonl` and settles a `DECISIONS.md` entry.

## Why Qwen-1.5B was used first (and why that is not the answer)

It was chosen for **speed of getting a pipeline standing up**, not on merit. A 1.5B model downloads in minutes, trains in ~20 minutes, and quantizes fast — which let the ingestion → RAG → serving path be proven end to end. It is the **control**, not the candidate. Selecting it without the comparison below would be exactly the popularity-driven shortcut `DECISIONS.md` exists to prevent.

## SLM-001 — Base model candidates

All candidates get the **same retrieved context** and the **same held-out questions**, on this machine. Hardware corrected 2026-09-18: **Intel Core Ultra 7 265, 31.5 GB RAM, NVIDIA RTX 2000 Ada 16 GB (SM 8.9, native bf16)** - see `docs/hardware_profile.md`. The "i7-12700H, 15.7 GB, CPU only" figures previously here described a different machine.

| Candidate | Params | Q4_K_M size | Fits 31.5 GB RAM? | Trainable on 16 GB VRAM |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.0 GB | Easily | **LoRA** (bf16 base ≈ 3 GB) — *control arm* |
| Llama-3.2-3B-Instruct | 3B | ~2.0 GB | Comfortably | **LoRA** (bf16 ≈ 6 GB) — *arm A*; also **QLoRA** as *arm C* |
| Mistral-7B-Instruct-v0.3 | 7B | ~4.4 GB | Yes, with margin | **QLoRA** — *arm B*; bf16 ≈ 15.4 GB vs 16.0 GB, will not fit with optimizer state |
| Llama-3.1-8B-Instruct | 8B | ~4.9 GB | Yes | **QLoRA** — not run in Phase 8 |

**The LoRA-vs-QLoRA comparison falls partly out of this table rather than being invented.** ≤3B can be LoRA-trained in bf16; 7B/8B cannot on 16 GB once optimizer state is counted. So method is partly determined by model, and the honest experiment is *(model, method)* pairs — not method in isolation. Arm C exists precisely so that one *(model, method)* pair is held fixed on the model side, isolating the method.

**Corrected 2026-09-18.** This table previously read "Fits 15.7 GB?" and "Trainable on free T4", which described the old laptop and Colab. Both are gone: training is local on a 16 GB RTX 2000 Ada, and the machine has 31.5 GB RAM. See `docs/hardware_profile.md` and `SLM-001`.

**The VRAM constraint is not the weights — it is the logits.** Cross-entropy materialises `batch × seq × vocab` at ~10 bytes per token per vocab entry. Llama-3.2-3B's **128,256**-token vocab against Mistral-7B-v0.3's **32,768** makes the 3B arm *tighter per token* than the 7B one, despite being under half the size.

Serving-side note: every candidate fits at Q4 with room to spare now. The constraint that actually bites on this machine is **disk** — 43.6 GB free, with 19.5 GB already held by the two HF caches — not RAM. Peak RSS is still measured with the whole stack loaded, not the model alone.

## FT-001 — Fine-tuning method

| Arm | Config | Purpose |
|---|---|---|
| **Control** | base model, no training | Without this, "fine-tuning helped" is unmeasurable |
| LoRA r=8 | bf16 base | Cheapest adapter |
| LoRA r=16 | bf16 base | Default |
| LoRA r=32 | bf16 base | Does more capacity help on 1.4k examples? |
| QLoRA r=16 | 4-bit NF4 base | Required for 7B/8B; also run at 1.5B to isolate the *quantized-base* effect from the *model-size* effect |

Same data, same splits, same epoch budget. Scored on: grounded correctness, **abstention rate on unanswerable questions**, role appropriateness, citation accuracy, and regression on general instruction-following.

## QUANT-001 — Quantization sweep

Run on the winning tuned model. `f16` is the reference; everything is measured as a delta from it.

| Build | Measure |
|---|---|
| f16 (reference) | quality baseline |
| Q8_0 | near-lossless check |
| Q6_K | |
| Q5_K_M | |
| Q4_K_M | |
| Q3_K_M | where quality visibly breaks |

Per build: file size · **peak RSS** · load time · first-token latency · tokens/sec · quality delta.

**Selection rule:** smallest build that stays within the quality threshold *and* meets interactive latency. Not the smallest file.

## RAG-001 — Retrieval variants

| Axis | Arms |
|---|---|
| Retriever | BM25 · dense (bge-small) · hybrid |
| Chunking | 1100 chars (current) · 600 · document/heading-aware |
| Reranker | none · local cross-encoder |
| Top-k | 3 · 5 · 8 |

Scored with Recall@k and MRR at document level, plus p50/p95 latency. `evaluation/eval_retrieval.py` runs this.

## Honest constraints on all of the above

1. **Free Colab T4 time is the bottleneck.** Four models × five FT arms is not a single afternoon. Priority order: (a) control vs LoRA on 1.5B, proving the training path works end to end; (b) Mistral-7B QLoRA, the strongest realistic candidate; (c) remaining arms as time allows.
2. **The current retrieval benchmark is auto-derived from passages**, which favours lexical overlap and flatters BM25. It can rank retrieval variants against each other, but human-written questions are still required before claiming anything about answer quality.
3. Each arm's raw numbers land in `experiments/*.jsonl` before any conclusion is written into `DECISIONS.md`.
