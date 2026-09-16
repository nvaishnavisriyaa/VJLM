"""Stage 5: build the fine-tuning dataset.

Teaches BEHAVIOUR, not facts (see data_catalog/classification.md).
Template-generated from real retrieved chunks - no hosted LLM, per INGEST-001.
Five parts: grounded answering, abstention, role differentiation,
VNR terminology, sensitive-topic tone.
"""
import json, os, random, re
random.seed(17)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(ROOT, "preprocessing", "out", "rag_chunks.jsonl")
OUT = os.path.join(ROOT, "training", "data")

SYS = ("You are VJLM, the Vignana Jyothi Language Model - the AI assistant for "
       "VNR Vignana Jyothi Institute of Engineering & Technology (VNRVJIET). "
       "Answer only from the CONTEXT provided. Cite the source. If the context "
       "does not contain the answer, say so plainly. Never invent dates, names, "
       "fees, or policies.")

ABSTAIN = ("I could not find this in the available VNRVJIET sources. "
           "Please check with the concerned department or the official website.")

# Questions whose answers are genuinely NOT on the public website
# (columns 2 and 3 of the data-source figure).
OUT_OF_SCOPE = [
    "What are my internal marks this semester?",
    "How many classes have I attended so far?",
    "Which faculty is free right now?",
    "What is today's canteen menu?",
    "Which companies are visiting campus next week?",
    "What is my exam hall and seat number?",
    "Can you tell me my CGPA?",
    "Which room is my class in today?",
    "What are my pending fee dues?",
    "Has my scholarship application been approved?",
    "What is my backlog list?",
    "Which faculty teaches section B of this subject?",
    "Are there any last-minute class changes today?",
    "What is my personal timetable?",
    "Am I eligible for the next placement drive?",
]

TERMS = {
    "MEME scheme": "the Minor and Honors degree framework at VNRVJIET, which lets a B.Tech student earn an additional specialisation alongside their main branch",
    "CBCS": "the Choice-Based Credit System, under which a student earns credits from core, elective and open-elective courses rather than a fixed syllabus",
    "R25": "the VNRVJIET academic regulation effective from 2025 onwards, currently applicable to I and II year B.Tech students",
    "R22": "the VNRVJIET academic regulation currently in force for III and IV year B.Tech students",
    "PEO": "a Programme Educational Objective - a broad statement of what a graduate is expected to achieve in the first few years after graduating",
    "PO": "a Programme Outcome - a specific ability a student is expected to have at the time of graduation",
    "PSO": "a Programme Specific Outcome - an outcome particular to one department's programme",
    "IQAC": "the Internal Quality Assurance Cell, which coordinates quality benchmarks and accreditation work at VNRVJIET",
    "IDEA Lab": "the innovation and prototyping laboratory at VNRVJIET where students build hardware and product ideas",
    "ICC": "the Internal Complaints Committee, the statutory body that handles complaints of sexual harassment",
    "autonomous status": "VNRVJIET's authority to design its own curriculum, conduct its own examinations and declare its own results, while degrees are awarded by JNTUH",
    "condonation": "formal permission granted to a student with attendance below the required threshold to still appear for examinations, subject to the regulation's conditions",
}

SENSITIVE = [
    ("I am being ragged by seniors. What should I do?",
     "I'm sorry you're going through this - what you're describing is serious and you have the right to act on it. "
     "VNRVJIET has an Anti-Ragging Committee and an Anti-Ragging Cell, and ragging is prohibited under UGC regulations. "
     "Report it to the Anti-Ragging Committee, your HoD, or the Students' Welfare office. You can also use the "
     "national anti-ragging helpline. You do not have to handle this alone, and a complaint can be made confidentially."),
    ("A faculty member is making me uncomfortable. Who do I talk to?",
     "Thank you for telling me - this is something the institute takes seriously and there is a formal channel for it. "
     "VNRVJIET has an Internal Complaints Committee (ICC), which is the statutory body for complaints of harassment. "
     "You can approach the ICC directly, and the process is confidential. If you would prefer to speak to someone first, "
     "the Students' Welfare office or your mentor can guide you to the right person."),
    ("I have a grievance about my marks. What is the process?",
     "VNRVJIET has a Grievance Redressal Cell for exactly this. The usual route is to first raise it with the concerned "
     "faculty or HoD, and if it is not resolved, to file a formal grievance with the Grievance Redressal Cell. "
     "For examination-specific issues there is also a revaluation process with its own deadline - check the current "
     "examination notification for the applicable dates and fee."),
    ("I am feeling very stressed and low because of academics.",
     "I'm glad you said something, and I want to take that seriously. Academic pressure is real and you deserve support "
     "with it. VNRVJIET has a student counselling facility and a mentor system - your assigned faculty mentor is a "
     "reasonable first contact, and the counselling service is confidential. If you are in distress right now, please "
     "reach out to someone you trust or a helpline immediately rather than waiting."),
]


def ex(system, user, assistant):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def ctx_block(rows):
    return "\n\n".join("[{}] (source: {}, page {})\n{}".format(
        i + 1, r["url"], r["page"], r["text"][:700]) for i, r in enumerate(rows))


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if 40 < len(s.strip()) < 320]


QTPL = {
    "academic_regulations": ["What does the regulation say about {t}?", "As per VNRVJIET regulations, {t}?"],
    "academic_calendar":    ["When is {t} according to the academic calendar?"],
    "syllabus":             ["What does the syllabus specify about {t}?"],
    "examinations":         ["What are the examination rules regarding {t}?"],
    "placements":           ["What does the placement record show about {t}?"],
    "departments":          ["What information is published about {t}?"],
    "fees_scholarships":    ["What is stated about {t}?"],
    "admissions":           ["What is the admission procedure regarding {t}?"],
    "research":             ["What research information is published about {t}?"],
    "governance_grievance": ["What is the institute's stated policy on {t}?"],
    "facilities":           ["What facilities are described for {t}?"],
    "forms_procedures":     ["What is the procedure for {t}?"],
    "clubs_events":         ["What is published about {t}?"],
}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    good = [r for r in rows if r["category"] != "faculty" and len(r["text"]) > 350]
    random.shuffle(good)
    print("{} chunks, {} usable for templating".format(len(rows), len(good)))
    data = []

    # ---- 1. GROUNDED ANSWERING ------------------------------------------
    n_ground = 0
    for r in good:
        if n_ground >= 700:
            break
        ss = sentences(r["text"])
        tpls = QTPL.get(r["category"])
        if not ss or not tpls:
            continue
        ans = ss[0]
        topic = " ".join(ans.split()[:6]).rstrip(".,;:").lower()
        q = random.choice(tpls).format(t=topic)
        a = "{}\n\nSource: {} (page {})".format(ans, r["url"], r["page"])
        data.append(ex(SYS, "CONTEXT:\n{}\n\nQUESTION: {}".format(ctx_block([r]), q), a))
        n_ground += 1

    # ---- 2. ABSTENTION ---------------------------------------------------
    n_abs = 0
    for _ in range(400):
        q = random.choice(OUT_OF_SCOPE)
        distract = random.sample(good, k=2)
        data.append(ex(SYS, "CONTEXT:\n{}\n\nQUESTION: {}".format(ctx_block(distract), q), ABSTAIN))
        n_abs += 1

    # ---- 3. ROLE DIFFERENTIATION ----------------------------------------
    n_role = 0
    pool = [r for r in good if r["category"] in
            ("academic_regulations", "examinations", "syllabus", "academic_calendar")]
    for r in pool[:200]:
        ss = sentences(r["text"])
        if not ss:
            continue
        fact = ss[0]
        cb = ctx_block([r])
        stu = ("In simple terms: {}\n\nWhat this means for you as a student: check this against your own "
               "regulation batch, since the rule can differ between R18, R19, A19, R22 and R25.\n\n"
               "Source: {} (page {})").format(fact, r["url"], r["page"])
        fac = ("{}\n\nFor reference when advising students: confirm which regulation batch applies before "
               "applying this clause, as the provisions differ across regulation versions.\n\n"
               "Source: {} (page {})").format(fact, r["url"], r["page"])
        data.append(ex(SYS + "\nThe user is a STUDENT. Explain plainly and practically.",
                       "CONTEXT:\n{}\n\nQUESTION: Explain this rule.".format(cb), stu))
        data.append(ex(SYS + "\nThe user is FACULTY. Be precise and procedural.",
                       "CONTEXT:\n{}\n\nQUESTION: Explain this rule.".format(cb), fac))
        n_role += 2

    # ---- 4. VNR TERMINOLOGY ---------------------------------------------
    n_term = 0
    VQ = ["What is {k}?", "Can you explain {k}?", "What does {k} mean at VNRVJIET?", "Define {k}."]
    for k, v in TERMS.items():
        for tpl in VQ:
            data.append(ex(SYS, tpl.format(k=k), "{} is {}.".format(k, v)))
            n_term += 1
        data.append(ex(SYS, "A junior asked me about {} - how should I explain it?".format(k),
                       "You can explain it this way: {} is {}.".format(k, v)))
        n_term += 1

    # ---- 5. SENSITIVE-TOPIC TONE ----------------------------------------
    n_sens = 0
    for q, a in SENSITIVE:
        for _ in range(25):
            data.append(ex(SYS, q, a))
            n_sens += 1

    random.shuffle(data)
    n = len(data)
    ntest = max(50, n // 20)
    nval = max(50, n // 20)
    test, val, train = data[:ntest], data[ntest:ntest + nval], data[ntest + nval:]
    for name, part in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for d in part:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\ngrounded={} abstention={} role={} terminology={} sensitive={}".format(
        n_ground, n_abs, n_role, n_term, n_sens))
    print("TOTAL={}  ->  train={} val={} test={}".format(n, len(train), len(val), len(test)))
    print("wrote -> " + OUT)


if __name__ == "__main__":
    main()
