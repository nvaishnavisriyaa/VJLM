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

## FOUND-004 — Base model family to shortlist

- **Status:** proposed — NOT decided
- **Date:** 2026-09-16
- **Requirements:** Open weights, license compatible with academic/on-premise deployment, instruct-tuned, LoRA/QLoRA-trainable on Colab, llama.cpp/GGUF-convertible, runs in the RAM budget from FOUND-002.
- **Candidates:** Hugging Face open-weight instruct models in the practical local size range — Mistral and Llama families named as a starting point by the project owner, plus other current candidates to be identified by browser research.
- **Why not decided:** Per the Phase 7 gate, the base model is selected **after** the benchmark (Phase 5) and baseline RAG (Phase 6) exist, by running the held-out VJLM questions against each candidate with identical retrieved context on this hardware. Naming a model now would be exactly the popularity-driven choice the plan forbids.
- **Next step:** At Phase 7, browser-verify each candidate's official model card for license, context length, tokenizer/chat template, fine-tuning support and current llama.cpp/GGUF support. Record versions and URLs here.
