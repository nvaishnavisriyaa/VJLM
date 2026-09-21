# vjlm-llama3b-lora

VJLM fine-tuned adapter, arm **A** of the Phase 8 matrix. These are the
**final-step (step 150) weights** - the configuration every published number
was measured from.

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Base revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Method | LoRA (bf16 base) |
| Adapter size | 97 MB |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate / schedule | 0.0002, cosine |
| Epochs / optimizer steps | 2.0 / 150 |
| Effective batch | 16 (2 x 8) |
| Max sequence length | 1280 |
| Seed | 42 |
| Training precision | bf16 |
| Pad token | `<|finetune_right_pad_id|>` |
| Chat template | native |
| Peak VRAM (training) | 10.61 GB |
| Training wall-clock | 74.0 min |

## Measured behaviour

147 held-out items (87 answerable, 60 that must be refused), temperature 0,
max_new_tokens 320, seed 42, NF4 inference.

| metric | value |
|---|---|
| grounded_f1 | **0.7567** |
| 95% CI | [0.7166, 0.7959] |
| abstain_recall | 1.0000 |
| abstain_precision | 1.0000 |
| false_abstain | 0 / 87 |
| citation_rate | 0.5102 |

Raw row: `822c92ed` in
[`experiments/generation_results.jsonl`](../../experiments/generation_results.jsonl).

## Why the final step and not the lowest-loss checkpoint

Validation loss bottoms out **inside epoch 1** on every arm and then rises, so
the conventional move would be to early-stop and ship that checkpoint. We
scored it instead of assuming, and it is worse on the metrics that matter:

| | step 150 (published) | lowest-loss checkpoint | delta |
|---|---|---|---|
| grounded_f1 | 0.7567 | 0.6591 | -0.0976 |
| citation_rate | 0.5102 | 0.4286 | -0.0816 |

All three arms moved the same way. `eval_loss` is token-level cross-entropy
against the reference wording; grounded answering, abstention and citation are
what VJLM is graded on, and here they diverged. Full validation curve:

| step | eval_loss |
|---|---|
| 25 | 0.3506 |
| 50 | 0.3271 |
| 75 | 0.3374 |
| 100 | 0.3469 |
| 125 | 0.3623 |
| 150 | 0.3619 |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE, REV = "meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95"
tok = AutoTokenizer.from_pretrained("models/vjlm-llama3b-lora")
model = AutoModelForCausalLM.from_pretrained(
    BASE, revision=REV, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, "models/vjlm-llama3b-lora")   # NOT merged
model.eval()
```

Pin the revision. `main` moves, and an adapter applied to different base
weights than it was trained against is a different model.

For CPU deployment the adapter is merged and converted to GGUF - see
[`docs/finetuning.md`](../../docs/finetuning.md) and `QUANT-001`.

## Licence

Derived from `meta-llama/Llama-3.2-3B-Instruct`. The Llama 3.2 Community License carries a 700M-MAU clause and a "Built with Llama" attribution requirement, and those terms reach derivatives. Check it before redistributing.
