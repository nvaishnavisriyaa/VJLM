import os,re,json,random,time,subprocess,urllib.request,hashlib,concurrent.futures as cf
random.seed(17); T0=time.time()
W="https://www.vnrvjiet.ac.in"; UA="VJLM-research-crawler (VNRVJIET academic project)"
BASE="Qwen/Qwen2.5-1.5B-Instruct"; NPDF=120
def sh(c): print("$",c,flush=True); subprocess.run(c,shell=True)
def F(u,t=90): return urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":UA}),timeout=t).read()
print("[1] discover",flush=True)
sc=F(W+"/").decode("utf-8","replace")
bd=F(W+re.search(r'src="(/assets/index-[^"]+\.js)"',sc).group(1),240).decode("utf-8","replace")
A=set(re.findall(r'"(/__l5e/assets-v1/[^"]+\.pdf)"',bd)); print("   ",len(A),"pdfs on site")
C=[("academic_regulations",r"regulation|attendance|detention|grading|cgpa|\b[ar]\d{2}\b"),
("academic_calendar",r"calender|calendar|commencement"),("syllabus",r"syllab|curricul|scheme|meme|credit|elective"),
("examinations",r"exam|timetable|malpractice|revaluation"),("placements",r"placement|recruit|internship|ctc"),
("departments",r"nba|naac|nirf|peo|pso|department|annual-report"),("admissions",r"admission|eamcet|counsel|intake"),
("governance_grievance",r"grievance|anti-ragging|icc|iqac|committee|policy")]
P=re.compile(r"^[a-z]+(-[a-z]+){1,3}(-ka-\d{4}-\d{2})?$")
def cl(n):
    for c,p in C:
        if re.search(p,n): return c
pk=[]
for u in sorted(A):
    nm=u.rsplit("/",1)[-1][:-4].lower()
    if P.match(nm) and not re.search(r"\d",nm): continue
    c=cl(nm)
    if c: pk.append((W+u,c))
pk=pk[:NPDF]; print("   ",len(pk),"selected"); os.makedirs("/content/pdfs",exist_ok=True)
def g(it):
    u,c=it; p="/content/pdfs/"+hashlib.sha1(u.encode()).hexdigest()[:16]+".pdf"
    try:
        if not os.path.exists(p): open(p,"wb").write(F(u))
        return (p,u,c)
    except Exception: return None
with cf.ThreadPoolExecutor(12) as e: got=[r for r in e.map(g,pk) if r]
print("   ",len(got),"downloaded %.0fs"%(time.time()-T0))
print("[2] extract",flush=True); sh("pip install -q pypdf")
from pypdf import PdfReader
EM=re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"); PH=re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
def rd(t): return PH.sub("[phone redacted]",EM.sub("[email redacted]",t))
def nm_(t): return re.sub(r"\n\s*\n\s*\n+","\n\n",re.sub(r"[ \t]+"," ",t.replace(" "," "))).strip()
K=[]
for p,u,c in got:
    try: R=PdfReader(p)
    except Exception: continue
    for i,pg in enumerate(R.pages):
        t=nm_(pg.extract_text() or "")
        if len(t)<300: continue
        for j in range(0,len(t),1000):
            s=t[j:j+1000]
            if len(s)>=300: K.append({"text":rd(s),"url":u,"page":i+1,"category":c})
print("   ",len(K),"chunks"); assert len(K)>200
print("[3] dataset",flush=True)
SYS=("You are VJLM, the Vignana Jyothi Language Model - the AI assistant for VNR Vignana Jyothi "
"Institute of Engineering & Technology (VNRVJIET). Answer only from the CONTEXT provided. Cite the "
"source. If the context does not contain the answer, say so plainly. Never invent dates, names, fees, or policies.")
AB="I could not find this in the available VNRVJIET sources. Please check with the concerned department or the official website."
OOS=["What are my internal marks?","How many classes have I attended?","Which faculty is free now?",
"What is today's canteen menu?","Which companies visit next week?","What is my exam hall and seat number?",
"What is my CGPA?","Which room is my class in today?","What are my pending fee dues?",
"Has my scholarship been approved?","What is my backlog list?","What is my personal timetable?"]
TM={"MEME scheme":"the Minor and Honors framework at VNRVJIET, letting a B.Tech student earn an extra specialisation alongside their main branch",
"CBCS":"the Choice-Based Credit System, where a student earns credits from core, elective and open-elective courses",
"R25":"the VNRVJIET regulation effective from 2025, for I and II year B.Tech students",
"R22":"the VNRVJIET regulation in force for III and IV year B.Tech students",
"PEO":"a Programme Educational Objective - what a graduate should achieve in the first years after graduating",
"PO":"a Programme Outcome - an ability a student should have at graduation",
"PSO":"a Programme Specific Outcome - an outcome particular to one department's programme",
"IQAC":"the Internal Quality Assurance Cell, coordinating quality benchmarks and accreditation",
"IDEA Lab":"the innovation and prototyping lab where students build hardware and product ideas",
"ICC":"the Internal Complaints Committee, the statutory body handling harassment complaints",
"condonation":"permission for a student below the attendance threshold to still sit examinations, subject to the regulation"}
SN=[("I am being ragged by seniors. What should I do?","I'm sorry you're going through this - what you describe is serious and you have the right to act on it. VNRVJIET has an Anti-Ragging Committee and Cell, and ragging is prohibited under UGC regulations. Report it to the Anti-Ragging Committee, your HoD, or Students' Welfare. A complaint can be made confidentially, and you do not have to handle this alone."),
("A faculty member is making me uncomfortable. Who do I talk to?","Thank you for telling me - the institute takes this seriously and there is a formal channel. VNRVJIET has an Internal Complaints Committee (ICC), the statutory body for harassment complaints. You can approach the ICC directly and the process is confidential. If you would rather speak to someone first, Students' Welfare or your mentor can guide you."),
("I have a grievance about my marks. What is the process?","VNRVJIET has a Grievance Redressal Cell for this. Raise it first with the concerned faculty or HoD; if unresolved, file a formal grievance with the Cell. For examination issues there is also a revaluation process with its own deadline and fee - check the current examination notification."),
("I am feeling very stressed and low because of academics.","I'm glad you said something. Academic pressure is real and you deserve support with it. VNRVJIET has student counselling and a faculty mentor system - your mentor is a reasonable first contact and counselling is confidential. If you are in distress right now, please reach out to someone you trust or a helpline rather than waiting.")]
QT={"academic_regulations":"What does the regulation say about {t}?","academic_calendar":"When is {t} per the academic calendar?",
"syllabus":"What does the syllabus specify about {t}?","examinations":"What are the examination rules regarding {t}?",
"placements":"What does the placement record show about {t}?","departments":"What is published about {t}?",
"admissions":"What is the admission procedure regarding {t}?","governance_grievance":"What is the stated policy on {t}?"}
def E(s,u,a): return {"messages":[{"role":"system","content":s},{"role":"user","content":u},{"role":"assistant","content":a}]}
def CX(rs): return "\n\n".join("[%d] (source: %s, page %d)\n%s"%(i+1,r["url"],r["page"],r["text"][:700]) for i,r in enumerate(rs))
def SE(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+",t) if 40<len(s.strip())<300]
random.shuffle(K); D=[]; ng=0
for r in K:
    if ng>=700: break
    ss=SE(r["text"])
    if not ss or r["category"] not in QT: continue
    a=ss[0]; q=QT[r["category"]].format(t=" ".join(a.split()[:6]).rstrip(".,;:").lower())
    D.append(E(SYS,"CONTEXT:\n%s\n\nQUESTION: %s"%(CX([r]),q),"%s\n\nSource: %s (page %d)"%(a,r["url"],r["page"]))); ng+=1
na=0
for _ in range(400):
    D.append(E(SYS,"CONTEXT:\n%s\n\nQUESTION: %s"%(CX(random.sample(K,2)),random.choice(OOS)),AB)); na+=1
nr=0; PL=[c for c in K if c["category"] in ("academic_regulations","examinations","syllabus","academic_calendar")]
for r in PL[:180]:
    ss=SE(r["text"])
    if not ss: continue
    f,cb=ss[0],CX([r])
    D.append(E(SYS+"\nThe user is a STUDENT. Explain plainly and practically.","CONTEXT:\n%s\n\nQUESTION: Explain this rule."%cb,"In simple terms: %s\n\nCheck this against your own regulation batch - the rule differs between R18, R19, A19, R22 and R25.\n\nSource: %s (page %d)"%(f,r["url"],r["page"])))
    D.append(E(SYS+"\nThe user is FACULTY. Be precise and procedural.","CONTEXT:\n%s\n\nQUESTION: Explain this rule."%cb,"%s\n\nWhen advising students, confirm which regulation batch applies - provisions differ across versions.\n\nSource: %s (page %d)"%(f,r["url"],r["page"]))); nr+=2
nt=0
for k,v in TM.items():
    for tp in ["What is %s?","Can you explain %s?","What does %s mean at VNRVJIET?","Define %s."]:
        D.append(E(SYS,tp%k,"%s is %s."%(k,v))); nt+=1
ns=0
for q,a in SN:
    for _ in range(25): D.append(E(SYS,q,a)); ns+=1
random.shuffle(D); n=len(D); k=max(40,n//20)
for nm2,pt in (("train",D[2*k:]),("val",D[k:2*k]),("test",D[:k])):
    open("/content/%s.jsonl"%nm2,"w",encoding="utf-8").write("".join(json.dumps(d,ensure_ascii=False)+"\n" for d in pt))
print("    grounded=%d abstain=%d role=%d terms=%d sens=%d TOTAL=%d"%(ng,na,nr,nt,ns,n))
print("[4] train LoRA",flush=True)
sh("pip uninstall -y -q torchao"); sh("pip install -q peft trl bitsandbytes")
import torch,inspect
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,PeftModel
from trl import SFTTrainer,SFTConfig
assert torch.cuda.is_available(),"Runtime > Change runtime type > T4 GPU"
print("    GPU:",torch.cuda.get_device_name(0))
tk=AutoTokenizer.from_pretrained(BASE)
if tk.pad_token is None: tk.pad_token=tk.eos_token
ds=load_dataset("json",data_files={"train":"/content/train.jsonl","validation":"/content/val.jsonl"})
ds=ds.map(lambda b:{"text":[tk.apply_chat_template(m,tokenize=False) for m in b["messages"]]},batched=True,remove_columns=["messages"])
md=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map="auto"); md.config.use_cache=False
pc=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
want=dict(output_dir="/content/vjlm-lora",num_train_epochs=2,per_device_train_batch_size=4,
gradient_accumulation_steps=4,learning_rate=2e-4,lr_scheduler_type="cosine",logging_steps=10,
save_strategy="epoch",save_total_limit=1,bf16=True,max_length=1024,report_to="none")
okp=set(inspect.signature(SFTConfig.__init__).parameters); al={"max_length":"max_seq_length"}; kw={}
for a_,b_ in want.items():
    if a_ in okp: kw[a_]=b_
    elif al.get(a_) in okp: kw[al[a_]]=b_
    else: print("    dropped",a_)
tp=set(inspect.signature(SFTTrainer.__init__).parameters)
tkw={"processing_class":tk} if "processing_class" in tp else {"tokenizer":tk}
tr=SFTTrainer(model=md,args=SFTConfig(**kw),train_dataset=ds["train"],eval_dataset=ds["validation"],peft_config=pc,**tkw)
tr.train(); tr.save_model("/content/vjlm-lora"); tk.save_pretrained("/content/vjlm-lora")
print("    trained %.1f min"%((time.time()-T0)/60))
print("[5] merge + quantize",flush=True)
del md,tr; torch.cuda.empty_cache()
bm=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16,device_map="cpu")
mg=PeftModel.from_pretrained(bm,"/content/vjlm-lora").merge_and_unload()
mg.save_pretrained("/content/vjlm-merged",safe_serialization=True); tk.save_pretrained("/content/vjlm-merged")
sh("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/lcpp")
sh("pip install -q -r /content/lcpp/requirements/requirements-convert_hf_to_gguf.txt")
sh("python /content/lcpp/convert_hf_to_gguf.py /content/vjlm-merged --outfile /content/vjlm-f16.gguf --outtype f16")
sh("cd /content/lcpp && cmake -B build -DLLAMA_CURL=OFF >/dev/null 2>&1 && cmake --build build --config Release -j --target llama-quantize >/dev/null 2>&1")
for q in ["Q4_K_M","Q5_K_M","Q8_0"]:
    sh("/content/lcpp/build/bin/llama-quantize /content/vjlm-f16.gguf /content/vjlm-%s.gguf %s"%(q,q))
sh("ls -lh /content/vjlm-*.gguf")
print("TOTAL %.1f min"%((time.time()-T0)/60))
from google.colab import files
files.download("/content/vjlm-Q4_K_M.gguf")
