# VJLM end-to-end on Colab: ingest -> dataset -> LoRA -> merge -> GGUF -> download
# Self-contained. No uploads. Mirrors the local pipeline exactly.
import os, re, json, random, time, subprocess, urllib.request, hashlib, concurrent.futures as cf
random.seed(17)
T0 = time.time()
WWW = "https://www.vnrvjiet.ac.in"
UA = "VJLM-research-crawler (VNRVJIET academic project)"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
NPDF = int(os.environ.get("VJLM_NPDF", "120"))


def sh(c):
    print("$", c, flush=True)
    subprocess.run(c, shell=True)


def fetch(url, t=90):
    r = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(r, timeout=t).read()


# ---------- 1. discover + fetch (same logic as ingestion/crawler) ----------
print("[1] discovering documents from the live site ...", flush=True)
shell = fetch(WWW + "/").decode("utf-8", "replace")
bundle_url = WWW + re.search(r'src="(/assets/index-[^"]+\.js)"', shell).group(1)
bundle = fetch(bundle_url, 240).decode("utf-8", "replace")
assets = set(re.findall(r'"(/__l5e/assets-v1/[^"]+\.pdf)"', bundle))
print("    %d PDFs referenced on www" % len(assets))

CATS = [("academic_regulations", r"regulation|attendance|detention|grading|cgpa|\b[ar]\d{2}\b"),
        ("academic_calendar", r"calender|calendar|commencement|almanac"),
        ("syllabus", r"syllab|curricul|scheme|meme|credit|elective"),
        ("examinations", r"exam|timetable|malpractice|revaluation"),
        ("placements", r"placement|recruit|internship|ctc"),
        ("departments", r"nba|naac|nirf|peo|pso|department|annual-report"),
        ("admissions", r"admission|eamcet|counsel|intake"),
        ("governance_grievance", r"grievance|anti-ragging|icc|iqac|committee|policy")]
PERSON = re.compile(r"^[a-z]+(-[a-z]+){1,3}(-ka-\d{4}-\d{2})?$")


def classify(name):
    for c, p in CATS:
        if re.search(p, name):
            return c
    return None


picked = []
for u in sorted(assets):
    nm = u.rsplit("/", 1)[-1][:-4].lower()
    if PERSON.match(nm) and not re.search(r"\d", nm):
        continue                      # skip faculty CVs (RAG only, never training)
    c = classify(nm)
    if c:
        picked.append((WWW + u, c))
picked = picked[:NPDF]
print("    selected %d institutional PDFs" % len(picked))

os.makedirs("/content/pdfs", exist_ok=True)


def grab(it):
    u, c = it
    p = "/content/pdfs/" + hashlib.sha1(u.encode()).hexdigest()[:16] + ".pdf"
    try:
        if not os.path.exists(p):
            open(p, "wb").write(fetch(u))
        return (p, u, c)
    except Exception:
        return None


with cf.ThreadPoolExecutor(12) as ex:
    got = [r for r in ex.map(grab, picked) if r]
print("    downloaded %d  (%.0fs)" % (len(got), time.time() - T0))

# ---------- 2. extract + chunk ----------
print("[2] extracting text ...", flush=True)
sh("pip install -q pypdf")
from pypdf import PdfReader


def norm(t):
    t = t.replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", t).strip()


EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def redact(t):
    return PHONE.sub("[phone redacted]", EMAIL.sub("[email redacted]", t))


chunks = []
for p, u, c in got:
    try:
        rd = PdfReader(p)
    except Exception:
        continue
    for i, pg in enumerate(rd.pages):
        t = norm(pg.extract_text() or "")
        if len(t) < 300:
            continue
        for j in range(0, len(t), 1000):
            s = t[j:j + 1000]
            if len(s) >= 300:
                chunks.append({"text": redact(s), "url": u, "page": i + 1, "category": c})
print("    %d chunks from %d docs" % (len(chunks), len(got)))
assert len(chunks) > 200, "too few chunks - site structure may have changed"

# ---------- 3. build the behavioural dataset ----------
print("[3] building fine-tuning dataset ...", flush=True)
SYS = ("You are VJLM, the Vignana Jyothi Language Model - the AI assistant for "
       "VNR Vignana Jyothi Institute of Engineering & Technology (VNRVJIET). "
       "Answer only from the CONTEXT provided. Cite the source. If the context "
       "does not contain the answer, say so plainly. Never invent dates, names, "
       "fees, or policies.")
ABSTAIN = ("I could not find this in the available VNRVJIET sources. "
           "Please check with the concerned department or the official website.")
OOS = ["What are my internal marks this semester?", "How many classes have I attended?",
       "Which faculty is free right now?", "What is today's canteen menu?",
       "Which companies are visiting next week?", "What is my exam hall and seat number?",
       "Can you tell me my CGPA?", "Which room is my class in today?",
       "What are my pending fee dues?", "Has my scholarship been approved?",
       "What is my backlog list?", "What is my personal timetable?",
       "Am I eligible for the next placement drive?"]
TERMS = {
    "MEME scheme": "the Minor and Honors degree framework at VNRVJIET, which lets a B.Tech student earn an additional specialisation alongside their main branch",
    "CBCS": "the Choice-Based Credit System, under which a student earns credits from core, elective and open-elective courses",
    "R25": "the VNRVJIET academic regulation effective from 2025, applicable to I and II year B.Tech students",
    "R22": "the VNRVJIET academic regulation in force for III and IV year B.Tech students",
    "PEO": "a Programme Educational Objective - what a graduate should achieve in the first few years after graduating",
    "PO": "a Programme Outcome - an ability a student is expected to have at graduation",
    "PSO": "a Programme Specific Outcome - an outcome particular to one department's programme",
    "IQAC": "the Internal Quality Assurance Cell, which coordinates quality benchmarks and accreditation",
    "IDEA Lab": "the innovation and prototyping laboratory where students build hardware and product ideas",
    "ICC": "the Internal Complaints Committee, the statutory body handling harassment complaints",
    "condonation": "permission for a student below the attendance threshold to still sit examinations, subject to the regulation",
}
SENS = [("I am being ragged by seniors. What should I do?",
         "I'm sorry you're going through this - what you describe is serious and you have the right to act on it. VNRVJIET has an Anti-Ragging Committee and Cell, and ragging is prohibited under UGC regulations. Report it to the Anti-Ragging Committee, your HoD, or Students' Welfare. A complaint can be made confidentially, and you do not have to handle this alone."),
        ("A faculty member is making me uncomfortable. Who do I talk to?",
         "Thank you for telling me - the institute takes this seriously and there is a formal channel. VNRVJIET has an Internal Complaints Committee (ICC), the statutory body for harassment complaints. You can approach the ICC directly and the process is confidential. If you would rather speak to someone first, Students' Welfare or your mentor can guide you."),
        ("I have a grievance about my marks. What is the process?",
         "VNRVJIET has a Grievance Redressal Cell for this. Raise it first with the concerned faculty or HoD; if unresolved, file a formal grievance with the Cell. For examination issues there is also a revaluation process with its own deadline and fee - check the current examination notification."),
        ("I am feeling very stressed and low because of academics.",
         "I'm glad you said something. Academic pressure is real and you deserve support with it. VNRVJIET has student counselling and a faculty mentor system - your mentor is a reasonable first contact and counselling is confidential. If you are in distress right now, please reach out to someone you trust or a helpline rather than waiting.")]

QT = {"academic_regulations": "What does the regulation say about {t}?",
      "academic_calendar": "When is {t} according to the academic calendar?",
      "syllabus": "What does the syllabus specify about {t}?",
      "examinations": "What are the examination rules regarding {t}?",
      "placements": "What does the placement record show about {t}?",
      "departments": "What information is published about {t}?",
      "admissions": "What is the admission procedure regarding {t}?",
      "governance_grievance": "What is the institute's stated policy on {t}?"}


def ex(sysm, user, asst):
    return {"messages": [{"role": "system", "content": sysm},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": asst}]}


def ctx(rs):
    return "\n\n".join("[%d] (source: %s, page %d)\n%s" % (i + 1, r["url"], r["page"], r["text"][:700])
                       for i, r in enumerate(rs))


def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if 40 < len(s.strip()) < 300]


random.shuffle(chunks)
data = []
ng = 0
for r in chunks:
    if ng >= 700:
        break
    ss = sents(r["text"])
    if not ss or r["category"] not in QT:
        continue
    a = ss[0]
    q = QT[r["category"]].format(t=" ".join(a.split()[:6]).rstrip(".,;:").lower())
    data.append(ex(SYS, "CONTEXT:\n%s\n\nQUESTION: %s" % (ctx([r]), q),
                   "%s\n\nSource: %s (page %d)" % (a, r["url"], r["page"])))
    ng += 1
na = 0
for _ in range(400):
    data.append(ex(SYS, "CONTEXT:\n%s\n\nQUESTION: %s" % (ctx(random.sample(chunks, 2)), random.choice(OOS)), ABSTAIN))
    na += 1
nr = 0
pool = [c for c in chunks if c["category"] in ("academic_regulations", "examinations", "syllabus", "academic_calendar")]
for r in pool[:180]:
    ss = sents(r["text"])
    if not ss:
        continue
    f, cb = ss[0], ctx([r])
    data.append(ex(SYS + "\nThe user is a STUDENT. Explain plainly and practically.",
                   "CONTEXT:\n%s\n\nQUESTION: Explain this rule." % cb,
                   "In simple terms: %s\n\nCheck this against your own regulation batch - the rule differs between R18, R19, A19, R22 and R25.\n\nSource: %s (page %d)" % (f, r["url"], r["page"])))
    data.append(ex(SYS + "\nThe user is FACULTY. Be precise and procedural.",
                   "CONTEXT:\n%s\n\nQUESTION: Explain this rule." % cb,
                   "%s\n\nWhen advising students, confirm which regulation batch applies - provisions differ across versions.\n\nSource: %s (page %d)" % (f, r["url"], r["page"])))
    nr += 2
nt = 0
for k, v in TERMS.items():
    for tpl in ["What is %s?", "Can you explain %s?", "What does %s mean at VNRVJIET?", "Define %s."]:
        data.append(ex(SYS, tpl % k, "%s is %s." % (k, v)))
        nt += 1
ns = 0
for q, a in SENS:
    for _ in range(25):
        data.append(ex(SYS, q, a))
        ns += 1

random.shuffle(data)
n = len(data)
k = max(40, n // 20)
test, val, train = data[:k], data[k:2 * k], data[2 * k:]
for nm, part in (("train", train), ("val", val), ("test", test)):
    with open("/content/%s.jsonl" % nm, "w", encoding="utf-8") as f:
        for d in part:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("    grounded=%d abstain=%d role=%d terms=%d sensitive=%d" % (ng, na, nr, nt, ns))
print("    TOTAL=%d -> train=%d val=%d test=%d" % (n, len(train), len(val), len(test)))

# ---------- 4. LoRA fine-tune ----------
print("[4] training (LoRA, bf16 base) ...", flush=True)
sh("pip uninstall -y -q torchao")
sh("pip install -q peft trl bitsandbytes")
import torch, inspect
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

assert torch.cuda.is_available(), "Runtime > Change runtime type > T4 GPU"
print("    GPU:", torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = load_dataset("json", data_files={"train": "/content/train.jsonl",
                                      "validation": "/content/val.jsonl"})
ds = ds.map(lambda b: {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
            batched=True, remove_columns=["messages"])

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

want = dict(output_dir="/content/vjlm-lora", num_train_epochs=2,
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            learning_rate=2e-4, lr_scheduler_type="cosine", logging_steps=10,
            save_strategy="epoch", save_total_limit=1, bf16=True,
            max_length=1024, report_to="none")
ok = set(inspect.signature(SFTConfig.__init__).parameters)
alias = {"max_length": "max_seq_length"}
cfg_kw = {}
for kk, vv in want.items():
    if kk in ok:
        cfg_kw[kk] = vv
    elif alias.get(kk) in ok:
        cfg_kw[alias[kk]] = vv
    else:
        print("    dropped arg:", kk)
cfg = SFTConfig(**cfg_kw)

tp = set(inspect.signature(SFTTrainer.__init__).parameters)
tkw = {"processing_class": tok} if "processing_class" in tp else {"tokenizer": tok}
tr = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                eval_dataset=ds["validation"], peft_config=peft_cfg, **tkw)
tr.train()
tr.save_model("/content/vjlm-lora")
tok.save_pretrained("/content/vjlm-lora")
print("    trained in %.1f min" % ((time.time() - T0) / 60))

# ---------- 5. merge + quantize ----------
print("[5] merging + quantizing ...", flush=True)
del model, tr
torch.cuda.empty_cache()
b = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cpu")
m = PeftModel.from_pretrained(b, "/content/vjlm-lora").merge_and_unload()
m.save_pretrained("/content/vjlm-merged", safe_serialization=True)
tok.save_pretrained("/content/vjlm-merged")

sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp")
sh("pip install -q -r /content/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
sh("python /content/llama.cpp/convert_hf_to_gguf.py /content/vjlm-merged --outfile /content/vjlm-f16.gguf --outtype f16")
sh("cd /content/llama.cpp && cmake -B build -DLLAMA_CURL=OFF > /dev/null 2>&1 && cmake --build build --config Release -j --target llama-quantize > /dev/null 2>&1")
for q in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
    sh("/content/llama.cpp/build/bin/llama-quantize /content/vjlm-f16.gguf /content/vjlm-%s.gguf %s" % (q, q))
sh("ls -lh /content/vjlm-*.gguf")
print("TOTAL %.1f min" % ((time.time() - T0) / 60))

from google.colab import files
files.download("/content/vjlm-Q4_K_M.gguf")
