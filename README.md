# VJLM — Vignana Jyothi Language Model

A local, role-aware, domain-specific AI assistant for VNRVJIET.

## Non-negotiables

| Constraint | Rule |
|---|---|
| Runtime | 100% local / on-premise inference. **No external LLM API at runtime.** |
| Phase 1 users | Students and faculty only. |
| Phase 1 data | Publicly available VNRVJIET website data only. No login-only or private institutional data. |
| Knowledge freshness | RAG must be self-maintaining: detect new, changed, moved and deleted content and sync the local knowledge base. |
| Security | Authentication/RBAC enforces access **before** retrieval. The model is never the security boundary. |
| Selection rule | No component is chosen because it is popular. Every choice is backed by a VJLM-specific comparison recorded in `DECISIONS.md`. |

## The plan

The full 15-phase implementation and decision plan is [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md), with the architecture and data-source figures in [`docs/figures/`](docs/figures/). Current progress against it is in [`HANDOFF.md`](HANDOFF.md).

Derived from the plan:

| Document | Covers |
|---|---|
| [`DECISIONS.md`](DECISIONS.md) | Every component choice, its evidence, rejected options, revisit trigger |
| [`data_catalog/classification.md`](data_catalog/classification.md) | Phase 2 — fine-tune vs RAG across all 14 public categories |
| [`experiments/PLAN.md`](experiments/PLAN.md) | Phases 6–9 — which models, FT methods, quantizations and retrievers get compared |
| [`docs/phase1_site_recon.md`](docs/phase1_site_recon.md) | Phase 1 — 9 measured findings about the VNRVJIET site |

## The three-way separation

- **Fine-tuning** — what VJLM inherently knows and how it behaves (stable VNR identity, terminology, role-specific answer style).
- **RAG** — what VJLM retrieves and verifies at query time (regulations, syllabi, calendars, faculty lists, placements — anything that changes).
- **RBAC** — what the authenticated user is allowed to retrieve.

Stable facts may live in both fine-tuning and RAG. Frequently changing facts must never be embedded into model weights as the primary source of truth.

## Where each workload runs

**Corrected 2026-09-18 — Colab has left the loop.** This section previously
sent fine-tuning to Google Colab because the target machine had no CUDA GPU
(`FOUND-003`). It now has one, so that condition no longer holds and every
workload is local.

| Workload | Where | Why |
|---|---|---|
| Ingestion, preprocessing, embedding, indexing | Local CPU | No GPU needed; data stays on-premise. |
| Fine-tuning (LoRA / QLoRA) | **Local GPU** — RTX 2000 Ada, 16 GB | SM 8.9, native bf16. `training/train_local.py` detects precision rather than assuming it. |
| Quantization (GGUF conversion) | Local | 31.5 GB RAM holds a 7B fp16 merge comfortably. |
| Inference / serving | Local CPU only | Hard requirement. |

Nothing in the serving path may call an external service. See
[`docs/hardware_profile.md`](docs/hardware_profile.md) for the machine every
number in `experiments/` was measured on.

## Layout

```
configs/         versioned configuration — no hard-coded paths or model names
data_catalog/    source classification: fine-tune vs RAG vs both vs neither
ingestion/       registry, crawler, raw snapshots, per-source sync state
preprocessing/   two pipelines: RAG chunking, and fine-tuning dataset building
rag/             embedding, index, retrieval, reranking
evaluation/      held-out VJLM benchmark: questions + gold sources
training/        fine-tuning datasets, Colab notebooks, adapters
quantization/    GGUF conversion + quality/speed/RAM measurements
auth/            authentication and RBAC policy
app/             backend API + student/faculty UI
experiments/     reproducible experiment runs and raw results
docs/            hardware profile, runbooks, phase deliverables
```

## Phase order

Phase 0 foundation -> 1 scope/access matrix -> 2 fine-tune vs RAG classification -> 3 dynamic ingestion -> 4 preprocessing -> **5 benchmark** -> 6 RAG baseline -> 7 base SLM selection -> 8 fine-tuning -> 9 quantization -> 10 auth/RBAC -> 11 role-aware retrieval -> 12 prompt builder -> 13 UI -> 14 final eval + ablations -> 15 maintenance.

Three gates that are easy to violate:

1. **The benchmark (Phase 5) is built before any component comparison.** The held-out split is never used to tune chunking, retrieval, model or quantization settings.
2. **No fine-tuning until baseline RAG retrieves correctly.** Most factual errors are retrieval errors; tuning the generator will not fix them.
3. **Base-model scores are recorded before training**, or fine-tuning gains cannot be measured.

## Status

Last updated 2026-09-18. Full detail in [`HANDOFF.md`](HANDOFF.md).

| Phase | State |
|---|---|
| 0–4 foundation → preprocessing | **Done.** 776/777 sources, 13,051 chunks (12,589 PDF + 462 page-derived) |
| 3 dynamic ingestion | **Done and automatic.** The pipeline runs when the app starts; no scheduler entry, no human step |
| 5 benchmark | **Not started.** `evaluation/questions.jsonl` and `gold_sources.jsonl` are still empty. Still the blocking item for *selection* |
| 6 RAG baseline | **Built, not decided.** BM25 + dense + hybrid all run; `RAG-001` cannot be settled until Phase 5 exists |
| 7 base SLM selection | **Proposed, not decided.** `SLM-001` has its first measured rows; `configs/models.yaml` stays null until the matrix completes |
| 8 fine-tuning | **Running.** Arm B (Mistral-7B QLoRA) trained and scored; arms A and C in progress |
| 9 quantization | Blocked on Phase 8 |
| 10–13 auth / retrieval / prompt / UI | Working, with authentication still a role simulation |
| 14–15 final eval, maintenance | Not started |

### Phase 8 — first tuned arm, measured

Mistral-7B-Instruct-v0.3 + QLoRA r=16, against the untouched NF4 base on the
same 147 held-out items, same seed, same decode settings. Raw rows in
[`experiments/generation_results.jsonl`](experiments/generation_results.jsonl).

| metric | base | tuned |
|---|---|---|
| grounded_f1 | 0.4401 `[0.3952, 0.4857]` | **0.7654** `[0.7192, 0.8073]` |
| false_abstain | 14 / 87 | **0** |
| abstain_precision | 0.8108 | **1.0000** |
| abstain_recall | 1.0000 | 1.0000 (held) |
| citation_rate | 0.3605 | **0.5102** |

The 95% bootstrap CIs on grounded F1 do not overlap. The gain is in the
direction that was actually hard: the base model already abstained on
everything it should, but *also* refused 14 answerable questions. Training
removed all 14 without breaking abstention.

**Read this as a fine-tuning result, not a Phase 5 result.** These 147 items
are the held-out split of the fine-tuning dataset, auto-derived from the
corpus like the retrieval benchmark. They can compare training arms against
each other on identical data, which is what Phase 8 needs. They cannot
establish that VJLM answers real student questions well.

**So the next thing that matters is still Phase 5.** Retrieval, the base model
and quantization all claim to be decided "by measurement", and every benchmark
that currently exists is auto-derived from the passages themselves — which
flatters BM25 and cannot judge answer quality at all.
