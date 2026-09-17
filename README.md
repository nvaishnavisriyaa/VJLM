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

## Split between local machine and Colab

| Workload | Where | Why |
|---|---|---|
| Ingestion, preprocessing, embedding, indexing | Local CPU | No GPU needed; data stays on-premise. |
| Fine-tuning (LoRA / QLoRA) | Google Colab GPU | No CUDA GPU on the target machine. |
| Quantization (GGUF conversion) | Colab or local | Whichever has the RAM headroom for the fp16 merge. |
| Inference / serving | Local CPU only | Hard requirement. |

Colab is a **build-time** tool. Nothing in the serving path may call it.

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

Phase 0 — foundation in place. Next: Phase 1 scope and public source catalog.
