# Target Hardware Profile

Required by the Phase 0 gate: *all benchmarks must be run on this machine, or the deviation must be noted in the experiment record.*

**Re-recorded 2026-09-18.** The profile below replaces the 2026-09-16 record, which described a **different machine** — an i7-12700H laptop with 15.7 GB RAM and no CUDA GPU. That is not the machine any current measurement was taken on, and every Phase 8 number in `experiments/generation_results.jsonl` comes from the machine described here. The superseded figures are kept at the bottom, because the Phase 0 gate is about traceability, and a benchmark recorded against the wrong hardware profile is worse than one with no profile at all.

## Development / training / inference machine

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 265 |
| Cores / threads | 20 physical / 20 logical (no SMT) |
| RAM | 31.5 GB |
| **GPU** | **NVIDIA RTX 2000 Ada Generation, 16.0 GB VRAM** |
| Compute capability | SM 8.9 — **native bf16** |
| NVIDIA driver | 596.71 |
| OS | Windows 11 Home Single Language, 10.0.26200, 64-bit |
| Disk | `C:` 474.4 GB total, **43.6 GB free** |
| Training Python | 3.10.11 (`.venv310`, `--system-site-packages`) |
| torch | 2.5.1+cu121 |
| Default `python` | 3.14.6 (conda) — **no torch wheels exist for it**; do not use it for training |
| Git | 2.55.0.windows.3 — not on `PATH` inside the venv shells |

## Training

**Local, on the GPU above.** Colab has left the loop. `FOUND-003` routed fine-tuning to Colab because no local CUDA GPU existed; one now does, so that condition no longer holds.

This also retires the fp16 advice in `HANDOFF.md` §4. That said *"use fp16 on a T4"* because Turing has no native bf16 and emulates it — a T4 workaround, not a general rule. SM 8.9 supports bf16 natively, and `training/train_local.py` **detects** precision rather than assuming it.

## What this constrains

The binding constraints have changed completely. RAM is no longer one of them.

**1. Disk is now the tightest resource — 43.6 GB free, not 522 GB.**
The two HF caches alone are 19.5 GB (Mistral-7B 13.5, Llama-3.2-3B 6.0). Adapter checkpoints add ~2.5 GB per 7B arm. Any Phase 9 merge writes a ~14.5 GB fp16 artifact before quantizing. Every script pre-flights `shutil.disk_usage`, because Windows degrades badly under ~5 GB free.

**2. VRAM is 16 GB, and the logits are what bind — not the weights.**
Cross-entropy materialises `batch × seq × vocab`, upcasts to fp32 and keeps a gradient: roughly 10 bytes per token per vocab entry. Llama-3.2-3B has a **128,256**-token vocab against Mistral-7B-v0.3's **32,768**, so the 3B arm is *tighter per token* than the 7B arm despite being under half the size. This is also why `per_device_eval_batch_size` must be set explicitly — it defaults to 8, which measured 34 GB "reserved" and 10.7 s/it on a 1.5B model.

**3. Windows/WDDM does not OOM — it silently degrades.**
CUDA spills into shared host memory rather than raising, so *a crash-free run is not proof it fitted*. Only `max_memory_allocated` is. `max_memory_reserved` routinely exceeds device VRAM on a perfectly healthy run because the caching allocator fragments and `expandable_segments` is **not supported on Windows** (tried, rejected).

**4. 31.5 GB RAM is ample, and is no longer the model-size ceiling.**
The old profile budgeted 6–8 GB for weights plus KV cache. That was a real constraint on a 15.7 GB laptop; it is not one here. A Phase 9 fp16 merge of a 7B model (~14.5 GB) fits in RAM comfortably.

**5. 20 physical cores make CPU generation viable**, but `runtime/start_llama.cmd` is still configured `-t 8` from the old machine and its fallback model path points at a file that does not exist here. Set thread count from measurement — more threads is not monotonically faster.

## Measurements to record at each model/quantization experiment

Model file size (GB) · peak VRAM allocated (GB) · peak RSS (GB) · load time (s) · first-token latency (s) · tokens/sec · quality delta vs the fp16 reference on the held-out VJLM benchmark.

Keep `eval_sec_per_answer_gpu` and `deployment_sec_per_answer_cpu` as **separate** fields. They are different measurements on different devices and must never be collapsed into one generic `sec_per_answer`.

## Open items

- [ ] Confirm free RAM under realistic load (browser + backend + index open), not at idle.
- [ ] Confirm the final deployment target. If VJLM is served from an institutional machine rather than this one, profile that machine too and re-run the Phase 9 quantization selection against it.
- [ ] `fastembed`, `onnxruntime`, `pypdf` and `playwright` are **not installed on this machine**, and `rag/index/` holds no built index. The committed chunks make BM25 rebuildable in seconds, but dense retrieval and the crawler cannot run here until those are installed. Needed before the Phase 9 end-to-end serving check.

---

## Superseded record (2026-09-16) — a different machine

Retained for traceability. **No current measurement was taken on this hardware.**

| Item | Value |
|---|---|
| CPU | 12th Gen Intel Core i7-12700H, 14 physical / 20 logical |
| RAM | 15.7 GB usable |
| GPU | Intel Iris Xe Graphics (integrated, ~1 GB, no CUDA) |
| Disk | `C:` 952.6 GB total, 522.6 GB free |
| Python | 3.12.1 |
| Git | 2.43.0.windows.1 |

Its conclusion — *"RAM is the binding constraint; budget 6–8 GB for the model; the iGPU is not a useful accelerator"* — was correct for that machine and is wrong for this one in every particular.
