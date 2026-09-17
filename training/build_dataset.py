"""Stage 5: build the fine-tuning dataset.

Teaches BEHAVIOUR, not facts (see data_catalog/classification.md).
Template-generated from real retrieved chunks - no hosted LLM, per INGEST-001.
Five parts: grounded answering, abstention, role differentiation,
VNR terminology, sensitive-topic tone.

Splitting discipline (MASTER_PLAN.md 7.2, classification.md "Rules carried
into Phase 8"): the train/val/test split is decided BEFORE any augmentation,
and by GROUP rather than by example. A source document, a terminology entry
and an out-of-scope question are each one group - every example derived from
that group lands in exactly one split. Augmentation (abstention context
resampling, sensitive-topic repeats) then runs on train only.

Splitting after augmentation is what previously put 32% of test answers
verbatim into train and made "the fine-tune helped" unmeasurable.
"""
import json, os, random, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NOTE: contact details from the public site are deliberately NOT redacted here
# (project owner's decision, 2026-09-17): all ingested data is publicly
# published by VNRVJIET. classification.md's "stripped in preprocessing" line
# describes the intent, not this file's behaviour - retrieval still redacts at
# prompt-build time via preprocessing.enrich.redact.
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

# Definitions only. A term earns a place in the weights when it is static AND
# not worth citing - vocabulary, not content. Anything that answers "which one
# applies to me" is a rolling fact and belongs in RAG, where it can be re-indexed
# and cited. See data_catalog/classification.md.
TERMS = {
    "MEME scheme": "the Minor and Honors degree framework at VNRVJIET, which lets a B.Tech student earn an additional specialisation alongside their main branch",
    "CBCS": "the Choice-Based Credit System, under which a student earns credits from core, elective and open-elective courses rather than a fixed syllabus",
    "R25": "the VNRVJIET academic regulation effective from the 2025 admission batch onwards; which year groups it currently covers moves every year, so check the regulation document for your batch",
    "R22": "the VNRVJIET academic regulation effective from the 2022 admission batch onwards; which year groups it currently covers moves every year, so check the regulation document for your batch",
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


# --- answer selection -------------------------------------------------------
# The previous revision took sentences(text)[0] blindly, which produced answers
# like "Hoek E ." and "2." - the splitter broke on initials, list markers and
# textbook citations - and letterheads, which repeat on every page. 56% of
# grounded answers failed a basic quality test. Three gates fix that: split
# correctly, only template over prose, and reject repeated boilerplate.

ABBR = {"no", "nos", "vol", "ed", "eds", "pp", "fig", "sec", "dr", "mr", "mrs",
        "ms", "prof", "sr", "jr", "st", "etc", "viz", "approx", "ltd", "pvt",
        "inc", "rs", "i.e", "e.g"}
BREAK = re.compile(r"[.!?]+\s+")
TAIL = re.compile(r"([A-Za-z.]+|\d+)\s*$")
SECTION = re.compile(
    r"^(UNIT|WEEK|MODULE|SEMESTER|S\.?\s?NO|TABLE|ANNEXURE|REFERENCES?|TEXT ?BOOKS?)\b", re.I)


def sentences(t):
    """Split on sentence ends, but never after an initial (David A.), a list
    marker (2.) or a known abbreviation."""
    out, start = [], 0
    for m in BREAK.finditer(t):
        pre = t[start:m.start()]
        tail = TAIL.search(pre)
        w = tail.group(1).lower().rstrip(".") if tail else ""
        if (len(w) == 1 and w.isalpha()) or w.isdigit() or w in ABBR:
            continue
        if pre.strip():
            out.append(pre.strip())
        start = m.end()
    if t[start:].strip():
        out.append(t[start:].strip())
    return out


def is_prose(t):
    """Reject tables, numbered lists and header blocks - a syllabus grid has no
    answerable sentence in it, however long it is."""
    letters = sum(c.isalpha() for c in t)
    if letters < 200 or sum(c.isdigit() for c in t) > letters * 0.25:
        return False
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if not lines:
        return False
    if sum(1 for l in lines if re.match(r"^\(?\d+[.)\]]?\s", l)) > len(lines) * 0.4:
        return False
    return sum(1 for l in lines if l.upper() == l) <= len(lines) * 0.5


def good_sentence(s):
    s = s.strip()
    if not (90 <= len(s) <= 320) or s.upper() == s:
        return False
    # A leading page number hides the section marker behind it ("24\n\nUNIT-II:").
    if SECTION.match(re.sub(r"^[\d\s.:-]+", "", s)):
        return False
    # A syllabus outline has no sentence punctuation anywhere - it is a course
    # topic list, not a statement, and it cannot be the answer to a question.
    if not re.search(r"[.!?]", s):
        return False
    if len(re.findall(r"[A-Za-z]{2,}", s)) < 14:
        return False
    return sum(c.isdigit() for c in s) <= len(s) * 0.15


def bkey(s):
    """Key a sentence by its opening - a letterhead is re-extracted slightly
    differently per document, so exact text under-counts it."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()[:60]


def boilerplate_index(chunks):
    """Boilerplate shows up across many documents AND, within one document, on
    every page. Either signal is enough (MASTER_PLAN 7.1)."""
    docs, hits = {}, Counter()
    for r in chunks:
        for s in sentences(r["text"]):
            if good_sentence(s):
                k = bkey(s)
                docs.setdefault(k, set()).add(r["source_id"])
                hits[k] += 1
    return {k for k in docs if len(docs[k]) > 2 or hits[k] > 3}


def pick_answer(text, boiler):
    for s in sentences(text):
        if good_sentence(s) and bkey(s) not in boiler:
            return s
    return None


TOPIC_STOP = set("the of and to in for a an is are was were be on at by with as from this that it its "
                 "shall will may can not no if then than which who what when where how their there "
                 "students student institute college".split())


def topic_of(sentence):
    """A question topic built from the sentence's distinctive terms rather than
    its opening words, so the question does not quote the answer's first line."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", sentence)
             if w.lower() not in TOPIC_STOP]
    seen, keep = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            keep.append(w.lower())
        if len(keep) == 5:
            break
    return " ".join(keep) if len(keep) >= 3 else None


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
    # The catch-all bucket from discover.py's classifier. Its prose is usable
    # but its category is not trusted, so the template asserts nothing.
    "general":              ["What does VNRVJIET publish about {t}?",
                             "What information is available about {t}?"],
}

# Grounded examples are capped per category so one large category cannot
# dominate the set (MASTER_PLAN 7.2, "do not over-represent one document type").
GROUNDED_PER_CATEGORY = 220


SPLITS = ("train", "val", "test")
# 10%, not 5%: the prose and boilerplate gates cut the usable yield per
# document, and at 5% the eval splits were starved - val ended up with zero
# role-differentiation examples, which makes that behaviour unmeasurable.
VAL_FRAC = TEST_FRAC = 0.10

GROUNDED_TOTAL = 700       # before splitting
ROLE_CHUNKS = 200          # each yields a student + faculty example
SENSITIVE_REPEATS = 25     # train only, after the split

# Abstention and terminology have few source groups (15 questions, 12 terms),
# so the default 5% would leave one or two eval examples. Abstention is the
# headline behavioural metric - abstain_recall needs a real denominator - so
# more GROUPS are held out, and each held-out question is then paired with
# several distinct held-out contexts to give the split usable size.
SMALL_GROUP_FRAC = 0.2
ABSTAIN_TRAIN_TOTAL = 400  # train only, after the split
ABSTAIN_EVAL_TOTAL = 60    # per eval split, over that split's held-out questions

ROLE_CATEGORIES = ("academic_regulations", "examinations", "syllabus", "academic_calendar")
VQ = ["What is {k}?", "Can you explain {k}?", "What does {k} mean at VNRVJIET?", "Define {k}."]


def assign_splits(keys, rng, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """Put every group key in exactly one split. Small group counts still get
    a val and a test member, so a category is never silently train-only."""
    keys = sorted(set(keys))
    rng.shuffle(keys)
    n = len(keys)
    ntest = max(1, int(round(n * test_frac)))
    nval = max(1, int(round(n * val_frac)))
    out = {}
    for i, k in enumerate(keys):
        out[k] = "test" if i < ntest else ("val" if i < ntest + nval else "train")
    return out


def quota(total):
    """Split a per-part budget across the three splits."""
    return {"train": int(total * (1 - VAL_FRAC - TEST_FRAC)),
            "val": int(total * VAL_FRAC),
            "test": int(total * TEST_FRAC)}


def check_leakage(data):
    """The split is the whole point of this file, so verify it rather than
    trusting it. ABSTAIN is a label, not a memorisable fact, so it is expected
    to recur across splits - the (question, answer) PAIR is what must not."""
    tr_pairs = {(e["messages"][1]["content"], e["messages"][2]["content"])
                for e in data["train"]}
    bad = 0
    for sp in ("val", "test"):
        n = sum(1 for e in data[sp]
                if (e["messages"][1]["content"], e["messages"][2]["content"]) in tr_pairs)
        print("  {:<5} examples identical to a train example: {}".format(sp, n))
        bad += n
    return bad


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(17)
    rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    cand = [r for r in rows if r["category"] != "faculty" and len(r["text"]) > 350]
    rng.shuffle(cand)
    good = [r for r in cand if is_prose(r["text"])]
    boiler = boilerplate_index(good)
    print("{} chunks -> {} candidates -> {} prose; {} boilerplate openings rejected".format(
        len(rows), len(cand), len(good), len(boiler)))

    # ---- 0. SPLIT FIRST, BY DOCUMENT ------------------------------------
    # One source document never appears in two splits, in any part. This also
    # covers the abstention distractor contexts, which are drawn per split.
    doc_split = assign_splits([r["source_id"] for r in good], rng)
    pool = {s: [] for s in SPLITS}
    for r in good:
        pool[doc_split[r["source_id"]]].append(r)
    print("documents  -> " + "  ".join(
        "{}={}".format(s, sum(1 for v in doc_split.values() if v == s)) for s in SPLITS))

    data = {s: [] for s in SPLITS}
    counts = {}

    # ---- 1. GROUNDED ANSWERING ------------------------------------------
    cap = quota(GROUNDED_TOTAL)
    n_ground = {s: 0 for s in SPLITS}
    per_cat = {s: Counter() for s in SPLITS}
    cat_used = Counter()
    for sp in SPLITS:
        for r in pool[sp]:
            if n_ground[sp] >= cap[sp]:
                break
            tpls = QTPL.get(r["category"])
            if not tpls or per_cat[sp][r["category"]] >= quota(GROUNDED_PER_CATEGORY)[sp]:
                continue
            ans = pick_answer(r["text"], boiler)
            if not ans:
                continue
            topic = topic_of(ans)
            if not topic:
                continue
            q = rng.choice(tpls).format(t=topic)
            a = "{}\n\nSource: {} (page {})".format(ans, r["url"], r["page"])
            data[sp].append(ex(SYS, "CONTEXT:\n{}\n\nQUESTION: {}".format(ctx_block([r]), q), a))
            n_ground[sp] += 1
            per_cat[sp][r["category"]] += 1
            cat_used[r["category"]] += 1
    counts["grounded"] = n_ground

    # ---- 2. ROLE DIFFERENTIATION ----------------------------------------
    # The student and faculty example from one chunk share a group: they are
    # near-identical and must never straddle the boundary.
    cap = quota(ROLE_CHUNKS)
    n_role = {s: 0 for s in SPLITS}
    for sp in SPLITS:
        for r in pool[sp]:
            if n_role[sp] >= cap[sp]:
                break
            if r["category"] not in ROLE_CATEGORIES:
                continue
            fact = pick_answer(r["text"], boiler)
            if not fact:
                continue
            cb = ctx_block([r])
            stu = ("In simple terms: {}\n\nWhat this means for you as a student: check this against your own "
                   "regulation batch, since the rule can differ between R18, R19, A19, R22 and R25.\n\n"
                   "Source: {} (page {})").format(fact, r["url"], r["page"])
            fac = ("{}\n\nFor reference when advising students: confirm which regulation batch applies before "
                   "applying this clause, as the provisions differ across regulation versions.\n\n"
                   "Source: {} (page {})").format(fact, r["url"], r["page"])
            data[sp].append(ex(SYS + "\nThe user is a STUDENT. Explain plainly and practically.",
                               "CONTEXT:\n{}\n\nQUESTION: Explain this rule.".format(cb), stu))
            data[sp].append(ex(SYS + "\nThe user is FACULTY. Be precise and procedural.",
                               "CONTEXT:\n{}\n\nQUESTION: Explain this rule.".format(cb), fac))
            n_role[sp] += 1
    counts["role"] = {s: n_role[s] * 2 for s in SPLITS}

    # ---- 3. ABSTENTION ---------------------------------------------------
    # Group = the out-of-scope question. AUGMENTATION (resampling distractor
    # contexts up to the target count) happens here, after the split, on train.
    oos_split = assign_splits(OUT_OF_SCOPE, rng, SMALL_GROUP_FRAC, SMALL_GROUP_FRAC)
    n_abs = {s: 0 for s in SPLITS}
    for sp in SPLITS:
        qs = [q for q in OUT_OF_SCOPE if oos_split[q] == sp]
        if not qs or not pool[sp]:
            continue
        target = ABSTAIN_TRAIN_TOTAL if sp == "train" else ABSTAIN_EVAL_TOTAL
        for i in range(target):
            distract = rng.sample(pool[sp], k=min(2, len(pool[sp])))
            data[sp].append(ex(SYS, "CONTEXT:\n{}\n\nQUESTION: {}".format(
                ctx_block(distract), qs[i % len(qs)]), ABSTAIN))
            n_abs[sp] += 1
    counts["abstention"] = n_abs

    # ---- 4. VNR TERMINOLOGY ---------------------------------------------
    # Group = the term. All five phrasings of one term stay together.
    term_split = assign_splits(list(TERMS), rng, SMALL_GROUP_FRAC, SMALL_GROUP_FRAC)
    n_term = {s: 0 for s in SPLITS}
    for k, v in TERMS.items():
        sp = term_split[k]
        for tpl in VQ:
            data[sp].append(ex(SYS, tpl.format(k=k), "{} is {}.".format(k, v)))
            n_term[sp] += 1
        data[sp].append(ex(SYS, "A junior asked me about {} - how should I explain it?".format(k),
                           "You can explain it this way: {} is {}.".format(k, v)))
        n_term[sp] += 1
    counts["terminology"] = n_term

    # ---- 5. SENSITIVE-TOPIC TONE ----------------------------------------
    # Group = the question. The x25 repeat is augmentation: train only.
    sens_split = assign_splits([q for q, _ in SENSITIVE], rng)
    n_sens = {s: 0 for s in SPLITS}
    for q, a in SENSITIVE:
        sp = sens_split[q]
        for _ in range(SENSITIVE_REPEATS if sp == "train" else 1):
            data[sp].append(ex(SYS, q, a))
            n_sens[sp] += 1
    counts["sensitive"] = n_sens

    for sp in SPLITS:
        rng.shuffle(data[sp])
        with open(os.path.join(OUT, sp + ".jsonl"), "w", encoding="utf-8") as f:
            for d in data[sp]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\ngrounded examples by category")
    for c, n in cat_used.most_common():
        print("   {:<24} {}".format(c, n))
    thin = [c for c in QTPL if cat_used[c] < 20]
    if thin:
        print("   thin or absent: " + ", ".join(sorted(thin)))
        print("   -> those PDFs are tables/posters, not prose. Templating cannot\n"
              "      reach them; they need human-authored examples (classification.md).")

    print("\n{:<14} {:>7} {:>7} {:>7}".format("part", *SPLITS))
    for part, c in counts.items():
        print("{:<14} {:>7} {:>7} {:>7}".format(part, *(c[s] for s in SPLITS)))
    print("{:<14} {:>7} {:>7} {:>7}".format(
        "TOTAL", *(len(data[s]) for s in SPLITS)))

    print("\nleakage check")
    bad = check_leakage(data)
    if bad:
        raise SystemExit("LEAKAGE: {} val/test examples duplicate a train example".format(bad))
    print("  clean - no val/test example duplicates a train example")
    if min(n_sens["test"], n_term["test"]) < 3:
        print("\nNOTE: sensitive/terminology have few source groups, so their test\n"
              "      coverage is thin by construction. Tone quality needs human\n"
              "      review, not this split.")
    print("\nwrote -> " + OUT)


if __name__ == "__main__":
    main()
