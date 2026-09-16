# Target Hardware Profile

Recorded 2026-09-16. Required by the Phase 0 gate. All benchmarks must be run on this machine, or the deviation must be noted in the experiment record.

## Development / inference machine

| Item | Value |
|---|---|
| CPU | 12th Gen Intel Core i7-12700H |
| Cores / threads | 14 physical / 20 logical |
| Base clock | 2.30 GHz |
| RAM | 15.7 GB usable |
| GPU | Intel Iris Xe Graphics (integrated, ~1 GB, **no CUDA**) |
| OS | Windows 11 Home Single Language, 10.0.26200, 64-bit |
| Disk | `C:` 952.6 GB total, 522.6 GB free |
| Python | 3.12.1 |
| Git | 2.43.0.windows.1 |

## Training

No local CUDA GPU. Fine-tuning runs on **Google Colab** (see `FOUND-003` in `DECISIONS.md`); weights are downloaded and quantized for local CPU serving.

## What this constrains

**RAM is the binding constraint, not disk and not compute.**

- 15.7 GB total. Windows, a browser and the app backend will be resident alongside the model. Budget roughly **6–8 GB** for model weights plus KV cache, and verify with measured peak RSS in Phase 9 rather than trusting file size.
- The embedding model, vector index and reranker (if one is justified) also sit in RAM during a query. Their footprint counts against the same budget and must be measured together with the generator, not separately.
- 14 physical cores make CPU generation viable. Set llama.cpp thread count from measurement — more threads is not monotonically faster, and hyperthread oversubscription usually hurts.
- The iGPU is not a useful accelerator here and must not be assumed in any plan.

## Measurements to record at each model/quantization experiment

Model file size (GB) · peak RSS (GB) · load time (s) · first-token latency (s) · tokens/sec · quality delta vs the fp16 reference on the held-out VJLM benchmark.

## Open items

- [ ] Confirm free RAM under realistic load (browser + backend + index open), not at idle.
- [ ] Confirm the final deployment target. If VJLM will be served from an institutional machine rather than this laptop, profile that machine too and re-run the Phase 9 quantization selection against it.
