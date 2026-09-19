# Fine-tuning — models, method, artifacts, results

Where the trained models live, what they are, how they were trained, and what
they scored. Written 2026-09-19, covering Phase 8.

Authoritative sources, if this file and they ever disagree — **believe them, not this**:

| Fact | Source of truth |
|---|---|
| Scores | [`experiments/generation_results.jsonl`](../experiments/generation_results.jsonl) |
| Training config actually used | `run.json` inside each adapter directory |
| Why each choice was made | [`DECISIONS.md`](../DECISIONS.md) — `SLM-001`, `FT-002`, `FT-003`, `FT-004`, `FT-005` |
| Per-answer outputs | `experiments/artifacts/<run_id>.jsonl` |

---

## 1. Where the models are

**Nothing here is in git.** `.gitignore` excludes `training/adapters/**`, and the
base models are in the Hugging Face cache. The repository holds the *recipe* and
the *results*; the weights are local to this machine. Rebuild them with
`training/run_matrix.py` (§6).

### Trained adapters — `C:\Users\E028.10\code\VJLM\training\adapters\`

| Directory | Arm | Size on disk | Final adapter | Checkpoints |
|---|---|---|---|---|
| `vjlm-qlora-mistral-7b-v0.3/` | B | 3.1 GB | `adapter_model.safetensors`, 161 MB | 25, 50, 75, 100, 125, 150 |
| `vjlm-lora-llama-3.2-3b/` | A | 1.9 GB | `adapter_model.safetensors`, 93 MB | 25, 50, 75, 100, 125, 150 |
| `vjlm-qlora-llama-3.2-3b/` | C | ~1.9 GB | `adapter_model.safetensors`, 93 MB | 25, 50, 75, 100, 125, 150 |

The directory is large relative to the adapter because every intermediate
checkpoint is kept, deliberately: `FT-004` explains that per-epoch saving with
`save_total_limit=1` would have discarded the evidence that every arm over-fits.

Each completed directory also holds `adapter_config.json`, `run.json` (the full
training record), the tokenizer files and `chat_template.jinja` — so an adapter
can be loaded without guessing which template or pad token it was trained with.

**These are LoRA adapters, not standalone models.** Each one is a set of low-rank
deltas that only means anything when applied on top of its exact base model at
the exact revision recorded in its `run.json`.

### Base models — Hugging Face cache, `C:\Users\E028.10\.cache\huggingface\hub\`

| Model | Cache size | Pinned revision |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | 14 GB | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| `meta-llama/Llama-3.2-3B-Instruct` | 6.0 GB | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| `Qwen/Qwen2.5-1.5B-Instruct` | 2.9 GB | *control arm from `FT-001`, not used in Phase 8* |

Revisions are pinned in `training/run_matrix.py` and recorded in every result
row, because `main` moves and "we used Mistral-7B" is not a reproducible claim.

### Not yet built

No merged fp16 model and no GGUF quantization exist. That is Phase 9
(`QUANT-001`), which the completed matrix now unblocks. `runtime/start_llama.cmd`
still points at a model path that does not exist on this machine.

---

## 2. What the models are

| | Arm B | Arm A | Arm C |
|---|---|---|---|
| Base | Mistral-7B-Instruct-v0.3 | Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct |
| Parameters | 7B | 3B | 3B |
| Vocabulary | 32,768 | 128,256 | 128,256 |
| Licence | Apache-2.0, ungated | Llama 3.2 Community (custom, gated) | same |
| Pad token | `<unk>` | `<|finetune_right_pad_id|>` | `<|finetune_right_pad_id|>` |
| Status | ✅ trained + scored | ✅ trained + scored | ✅ trained + scored |

**The licences differ in a way that matters.** Mistral is Apache-2.0. Llama-3.2
is gated behind a custom licence with a 700M-MAU clause and a "Built with Llama"
attribution requirement. If VJLM is ever distributed rather than run
on-premise, that is a live constraint — and since the 3B model currently
performs as well as the 7B (§5), the licence may decide this rather than the
scores. `Qwen2.5-3B-Instruct` (Apache-2.0, ungated) is the drop-in substitute.

**The vocabulary difference is not trivia.** It drives VRAM, because
cross-entropy materialises `batch × seq × vocab`. It is why the bf16 3B arm
peaked at *more* VRAM (10.61 GB) than the 7B one (7.18 GB), and why arm C took
three attempts to run — see `FT-005`.

---

## 3. Which technique

All three arms use **LoRA-family adapter tuning** — the base weights are
frozen and only small low-rank matrices are trained. Nothing does full
fine-tuning; it would not fit on 16 GB, and it would also be the wrong tool,
since `DECISIONS.md` assigns *facts* to RAG and only *behaviour* to training.

| | Arm A | Arm B | Arm C |
|---|---|---|---|
| Technique | **LoRA** | **QLoRA** | **QLoRA** |
| Base weights held as | bf16 | 4-bit NF4, double-quantized | 4-bit NF4, double-quantized |
| Trainable | LoRA adapters only | LoRA adapters only | LoRA adapters only |

Everything below is **identical across arms** by design, so that A↔C isolates
the method. From `run.json`, not from the plan:

| Setting | Value |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05, `bias="none"` |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` — explicit, never left to PEFT's default |
| Learning rate / schedule | 2e-4, cosine |
| Epochs / optimizer steps | 2 / 150 |
| Batch | 2 per device × 8 accumulation = **effective 16** |
| Max sequence length | 1280 |
| Seed | 42 (Python, NumPy, Torch) |
| Precision | bf16 (SM 8.9 native) |
| Chat template | each model's own native template |
| Gradient checkpointing | on |

### Loss masking

Only the assistant's answer is trained on. The collator turns
`completion_mask == 0` into label `-100`, so the system prompt, the user turn and
padding contribute no loss, while the genuine end-of-turn token stays trainable.

Labels are **never** masked by comparing against the pad token id. Where PAD and
EOS share an id, that would silently delete the model's only signal for when to
stop. `FT-002` records the gate that verifies this, and the two bugs it caught.

### What each arm's scoring configuration was

All arms are scored at **NF4 inference**, including arm A, which trained in
bf16. Inference quantization is held constant so it cannot act as a hidden
second variable. Generation is locked at `temperature=0`,
`max_new_tokens=320`, seed 42. Reasoning in `FT-003`.

---

## 4. Training data

| Split | Rows | File |
|---|---|---|
| train | 1,199 | `training/data/train.jsonl` |
| val | 147 | `training/data/val.jsonl` |
| test | 147 | `training/data/test.jsonl` |

Content-hashed and recorded in every run: train `2b34546041463e84`, val
`62e83f8fee4da45f`, test `54b5bd2aed9ca354`. Identical across all arms.

Nothing was truncated at 1280 tokens in any arm — longest example 1,094 tokens
under the Llama tokenizer, 1,139 under Mistral's.

The split is by **document group before augmentation**. An earlier split shared
32% of its test answers with training, which made every result meaningless; see
`HANDOFF.md` §2.

---

## 5. Results

147 held-out test items, of which **87 are answerable** and **60 are
questions the model should refuse**. Same items, same seed, same decode
settings for every row.

| | Mistral-7B base | **7B QLoRA** | Llama-3B base | **3B LoRA** | **3B QLoRA** |
|---|---|---|---|---|---|
| run_id | `e38613e2` | `baffeb06` | `b48d1ca6` | `822c92ed` | `0868f01f` |
| arm / role | B / base | B / tuned | AC / base | A / tuned | C / tuned |
| grounded_f1 | 0.4401 | **0.7654** | 0.2369 | **0.7567** | **0.7567** |
| 95% CI | 0.395–0.486 | 0.719–0.807 | 0.202–0.273 | 0.717–0.796 | 0.712–0.796 |
| abstain_recall | 1.0000 | 1.0000 | 0.2833 | **1.0000** | **1.0000** |
| abstain_precision | 0.8108 | **1.0000** | 0.5862 | **1.0000** | **1.0000** |
| false_abstain | 14 / 87 | **0** | 12 / 87 | **0** | **0** |
| citation_rate | 0.3605 | 0.5102 | 0.0204 | 0.5102 | 0.5170 |
| mean answer tokens | 79 | 35 | 44 | 36 | 35 |
| train minutes | — | 230.6 | — | **74.0** | 105.1 |
| peak VRAM | — | 7.18 GB | — | 10.61 GB | **4.88 GB** |
| s / answer (GPU eval) | 22.19 | 20.16 | 5.83 | 13.72 | **13.24** |

All rows: 147 requested, 147 succeeded, 0 failed.

### What these numbers say

**Fine-tuning worked, and it worked in the difficult direction.** Both base
models already refuse things — Mistral refuses *everything* it should,
`abstain_recall` 1.000 before any training. The hard error is the opposite one:
refusing questions that *were* answerable. Mistral base did that 14 times,
Llama base 12. Both tuned arms do it **zero** times, without breaking
abstention. A model can trivially reach `false_abstain = 0` by never refusing;
neither of these did.

**The 3B model is not measurably worse than the 7B one.** 0.7567 against
0.7654, with confidence intervals overlapping across nearly their whole width,
against a **pre-registered** noise floor of ~0.03 F1 — `SLM-001` fixed that
threshold before any arm ran, so it cannot have been chosen to fit the result.
The 3B arm trains 3.1× faster and answers 1.5× faster.

**Base-model choice mattered far less than training.** The two bases differ by
0.2032 F1; the two tuned arms by 0.0087. Llama-3B base is genuinely poor —
`abstain_recall` 0.283 means it answered 43 of the 60 questions it should have
refused, and it cited on 3 of 147 answers. After training it refuses all 60 and
cites on 75. Both arms converge on nearly identical behaviour from very
different starting points.

### Three traps in this table

**1. `citation_rate` cannot exceed 0.5918.** Its denominator is all 147 items,
but only 87 are answerable and an abstention has nothing to cite — 87/147 =
0.5918. So 0.5102 means **75 of 87 answerable items cited, 86% of what is
reachable**, not "barely half". Both tuned arms cite 75 items and agree on 74
of them, so the identical figure is convergence, not a copied number — checked
per-item, because two arms matching to four decimals is what a metrics bug
looks like. Mistral *base*, for its part, attached citations to 10 items it
abstained on; both tuned arms never do.

**2. Validation loss does not rank the arms.** Arm B reaches `eval_loss`
0.2349, arm A only 0.3271 — yet their grounded F1 is a tie. This is not a
paradox, it is a category error waiting to happen: **cross-entropy is not
comparable across different tokenizers.** Mistral's 32,768-token vocabulary and
Llama's 128,256 give per-token losses on different scales. `eval_loss` is valid
*within* one arm, across steps. Never between arms.

**3. Both arms over-fit, and neither best checkpoint is what was scored.**

| step | arm A `eval_loss` | arm B `eval_loss` | arm C `eval_loss` |
|---|---|---|---|
| 25 | 0.3506 | 0.2728 | 0.3468 |
| 50 | **0.3271 — best** | 0.2376 | **0.3230 — best** |
| 75 | 0.3374 | **0.2349 — best** | 0.3405 |
| 100 | 0.3469 | 0.2627 | 0.3590 |
| 125 | 0.3623 | 0.2706 | 0.3675 |
| 150 | 0.3619 | 0.2754 | 0.3683 |

Both Llama arms bottom at step 50 and Mistral at step 75 — all inside 1 epoch
of a 2-epoch budget — then rise for the rest of training. **The scored adapters are
the step-150 ones in every case**, because `trainer.save_model()` writes the
final step. They still beat their baselines decisively, so the conclusions
above hold, but "does one epoch beat two?" is unanswered and answerable: every
checkpoint is on disk, and it costs one eval pass per arm with no retraining.
Open item, not a decision.

### A↔C — the controlled comparison: LoRA vs QLoRA is a wash on quality

Arms A and C differ in exactly one respect: the base weights are bf16 for A and
4-bit NF4 for C. Data, order, seed, schedule, rank, alpha, dropout, target
modules, batch, effective batch, sequence length and chat template are all
identical.

| | A — LoRA | C — QLoRA |
|---|---|---|
| grounded_f1 | 0.756694 | 0.756702 |
| false_abstain | 0 | 0 |
| citation_rate | 0.5102 (75/87) | 0.5170 (76/87) |
| **peak VRAM** | 10.61 GB | **4.88 GB** — 54% less |
| **train minutes** | **74.0** | 105.1 — 42% more |

The two means agree to eight parts in a million. That is *not* a duplicated
row — different `run_id`, and **67 of 87 items score differently**; they happen
to average to the same place. 75 of 147 predictions are byte-identical, nearly
all abstentions, which are formulaic by design.

**The trade is VRAM against wall-clock, not quality.** NF4 weights are
dequantized on every forward and again in backward under gradient checkpointing,
and that compute costs 42% more time. Quantizing the base cost nothing
measurable in behaviour here.

Scope: one model, one rank, one dataset, 87 scorable items. This says QLoRA is
a sound choice *in this setting*. It does not establish that 4-bit training is
free in general.

**Arm C required an allocator fix to run at all.** It exceeded 16 GB at step 45
twice before the cause was found: the caching allocator was reserving 13.21 GB
to serve 6.18 GB of live tensors and never releasing it. The fix is allocator
configuration — `--alloc-guard` — and changes no hyperparameter, which is why
arm A did not need rerunning and A↔C stays controlled. Full account, including
two wrong diagnoses, in `FT-005`.

---

## 6. Reproducing this

```bash
.venv310/Scripts/python.exe -u training/run_matrix.py
```

Every stage is idempotent, so this is safe to re-run: a training stage is
complete when its `run.json` exists, a scoring stage when its `(arm, role)` row
is already in `generation_results.jsonl`. Re-running skips finished work in
under a second and continues from wherever it stopped. It should be launched
detached — `FT-004` explains why, at the cost of a lost 48-minute run.

Environment is pinned in `requirements-train.txt` (torch 2.5.1+cu121,
transformers 4.56.2, trl 0.23.0, peft 0.17.1, bitsandbytes 0.47.0) against
Python 3.10 in `.venv310`. Package versions are recorded in every run record
and every result row, along with the git sha.

Single arm, directly:

```bash
.venv310/Scripts/python.exe -u training/train_local.py --base meta-llama/Llama-3.2-3B-Instruct --name vjlm-lora-llama-3.2-3b --epochs 2 --resume
```

---

## 7. What this evaluation does not establish

**These 147 items are templated and auto-derived from the VJLM corpus**, the
same way the retrieval benchmark is. They share vocabulary with their source
documents, which flatters any system that pattern-matches.

They are enough to rank training arms against each other on identical data,
which is exactly what Phase 8 needs. They are **not** enough to claim VJLM
answers real student questions well. `evaluation/questions.jsonl` and
`gold_sources.jsonl` are still empty, and Phase 5 remains the blocker on any
real model selection.

For this reason `SLM-001` is still `proposed — NOT decided`, and
`configs/models.yaml` is still null. The language throughout is
**"best-performing model on the Phase 8 templated behavioural evaluation"** —
never "the winner".

Everything here was measured on the machine in
[`hardware_profile.md`](hardware_profile.md): RTX 2000 Ada, 16 GB VRAM,
31.5 GB RAM, Windows 11.
