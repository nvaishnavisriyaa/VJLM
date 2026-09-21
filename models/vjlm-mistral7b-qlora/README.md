# vjlm-mistral7b-qlora

VJLM fine-tuned adapter, arm **B** of the Phase 8 matrix. These are the
**final-step (step 150) weights** - the configuration every published number
was measured from.

| | |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Base revision | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Method | QLoRA (4-bit NF4 base, double-quantized) |
| Adapter size | 168 MB |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate / schedule | 0.0002, cosine |
| Epochs / optimizer steps | 2.0 / 150 |
| Effective batch | 16 (2 x 8) |
| Max sequence length | 1280 |
| Seed | 42 |
| Training precision | bf16 |
| Pad token | `<unk>` |
| Chat template | native |
| Peak VRAM (training) | 7.18 GB |
| Training wall-clock | 230.6 min |

## Measured behaviour

147 held-out items (87 answerable, 60 that must be refused), temperature 0,
max_new_tokens 320, seed 42, NF4 inference.

| metric | value |
|---|---|
| grounded_f1 | **0.7654** |
| 95% CI | [0.7192, 0.8073] |
| abstain_recall | 1.0000 |
| abstain_precision | 1.0000 |
| false_abstain | 0 / 87 |
| citation_rate | 0.5102 |

Raw row: `baffeb06` in
[`experiments/generation_results.jsonl`](../../experiments/generation_results.jsonl).

## Why the final step and not the lowest-loss checkpoint

Validation loss bottoms out **inside epoch 1** on every arm and then rises, so
the conventional move would be to early-stop and ship that checkpoint. We
scored it instead of assuming, and it is worse on the metrics that matter:

| | step 150 (published) | lowest-loss checkpoint | delta |
|---|---|---|---|
| grounded_f1 | 0.7654 | 0.7389 | -0.0265 |
| citation_rate | 0.5102 | 0.4966 | -0.0136 |

All three arms moved the same way. `eval_loss` is token-level cross-entropy
against the reference wording; grounded answering, abstention and citation are
what VJLM is graded on, and here they diverged. Full validation curve:

| step | eval_loss |
|---|---|
| 25 | 0.2728 |
| 50 | 0.2376 |
| 75 | 0.2349 |
| 100 | 0.2627 |
| 125 | 0.2706 |
| 150 | 0.2754 |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE, REV = "mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"
tok = AutoTokenizer.from_pretrained("models/vjlm-mistral7b-qlora")
model = AutoModelForCausalLM.from_pretrained(
    BASE, revision=REV, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, "models/vjlm-mistral7b-qlora")   # NOT merged
model.eval()
```

Pin the revision. `main` moves, and an adapter applied to different base
weights than it was trained against is a different model.

For CPU deployment the adapter is merged and converted to GGUF - see
[`docs/finetuning.md`](../../docs/finetuning.md) and `QUANT-001`.

## Licence

Derived from `mistralai/Mistral-7B-Instruct-v0.3`. Apache-2.0.
