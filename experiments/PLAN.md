# VJLM Experiment Plan — what gets compared, and against what

No component is selected without beating alternatives on the same benchmark, same hardware. Every run appends raw numbers to `experiments/*.jsonl` and settles a `DECISIONS.md` entry.

## Why Qwen-1.5B was used first (and why that is not the answer)

It was chosen for **speed of getting a pipeline standing up**, not on merit. A 1.5B model downloads in minutes, trains in ~20 minutes, and quantizes fast — which let the ingestion → RAG → serving path be proven end to end. It is the **control**, not the candidate. Selecting it without the comparison below would be exactly the popularity-driven shortcut `DECISIONS.md` exists to prevent.

## SLM-001 — Base model candidates

All candidates get the **same retrieved context** and the **same held-out questions**, on this machine (i7-12700H, 15.7 GB RAM, CPU only).

| Candidate | Params | Q4_K_M size | Fits 15.7 GB? | Trainable on free T4 |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.0 GB | Easily | **LoRA** (bf16 base ≈ 3 GB) |
| Llama-3.2-3B-Instruct | 3B | ~2.0 GB | Comfortably | **LoRA** (bf16 ≈ 6 GB) |
| Mistral-7B-Instruct-v0.3 | 7B | ~4.4 GB | Yes, with margin | **QLoRA** — bf16 ≈ 14 GB, will not fit with optimizer state |
| Llama-3.1-8B-Instruct | 8B | ~4.9 GB | Yes, tightest | **QLoRA** |

**The LoRA-vs-QLoRA comparison falls out of this table rather than being invented.** Small models can be LoRA-trained in bf16; 7B/8B cannot fit on a free T4 that way. So the method is partly determined by the model, and the honest experiment is *(model, method)* pairs — not method in isolation.

Serving-side note: every candidate fits 15.7 GB at Q4. The constraint that actually bites is **RAM shared with the embedder, index and browser**, so peak RSS is measured with the whole stack loaded, not the model alone.

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
