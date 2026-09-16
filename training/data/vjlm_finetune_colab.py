# VJLM — LoRA fine-tuning on Colab GPU, then GGUF quantization.
#
# Run this in ONE Colab cell (Runtime > Change runtime type > T4 GPU).
# Upload training/data/train.jsonl and val.jsonl first (left sidebar > Files).
#
# Trains BEHAVIOUR, not facts: grounded answering, abstention, role
# differentiation, VNR terminology, sensitive-topic tone.
# Base model stays swappable — change BASE below.

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUTDIR = "/content/vjlm-lora"
MERGED = "/content/vjlm-merged"
EPOCHS = 2
MAXLEN = 1024

import subprocess, sys, os, json, time

def sh(c):
    print("$", c, flush=True)
    subprocess.run(c, shell=True, check=True)

t0 = time.time()

# ---- 1. deps -------------------------------------------------------------
sh("pip -q install -U transformers peft trl datasets accelerate bitsandbytes")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "Set Runtime > Change runtime type > T4 GPU"

# ---- 2. data -------------------------------------------------------------
assert os.path.exists("/content/train.jsonl"), "Upload train.jsonl to /content first"
ds = load_dataset("json", data_files={"train": "/content/train.jsonl",
                                      "validation": "/content/val.jsonl"})
print(ds)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def fmt(b):
    return {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]}

ds = ds.map(fmt, batched=True, remove_columns=["messages"])
print("sample:\n", ds["train"][0]["text"][:600])

# ---- 3. model + LoRA -----------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ---- 4. train ------------------------------------------------------------
cfg = SFTConfig(
    output_dir=OUTDIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,          # Colab reclaims runtimes — keep checkpoints
    bf16=True,
    max_length=MAXLEN,
    report_to="none",
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                     eval_dataset=ds["validation"], peft_config=peft_cfg,
                     processing_class=tok)
trainer.train()
trainer.save_model(OUTDIR)
tok.save_pretrained(OUTDIR)
print("training done in %.1f min" % ((time.time() - t0) / 60))

# ---- 5. merge LoRA into base --------------------------------------------
from peft import PeftModel
del model, trainer
torch.cuda.empty_cache()

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(base, OUTDIR).merge_and_unload()
merged.save_pretrained(MERGED, safe_serialization=True)
tok.save_pretrained(MERGED)
print("merged ->", MERGED)

# ---- 6. GGUF convert + quantize -----------------------------------------
sh("git clone -q --depth 1 https://github.com/ggerganov/llama.cpp /content/llama.cpp || true")
sh("pip -q install -r /content/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
sh(f"python /content/llama.cpp/convert_hf_to_gguf.py {MERGED} "
   f"--outfile /content/vjlm-f16.gguf --outtype f16")

sh("cd /content/llama.cpp && cmake -B build -DLLAMA_CURL=OFF > /dev/null && "
   "cmake --build build --config Release -j --target llama-quantize > /dev/null")

for q in ["Q4_K_M", "Q5_K_M"]:
    sh(f"/content/llama.cpp/build/bin/llama-quantize /content/vjlm-f16.gguf "
       f"/content/vjlm-{q}.gguf {q}")

sh("ls -lh /content/vjlm-*.gguf")

# ---- 7. download ---------------------------------------------------------
from google.colab import files
print("TOTAL %.1f min" % ((time.time() - t0) / 60))
print("Downloading Q4_K_M (the one that runs on the target laptop)...")
files.download("/content/vjlm-Q4_K_M.gguf")
