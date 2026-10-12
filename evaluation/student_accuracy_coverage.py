"""Student-persona accuracy test: role=student, mode=coverage, live app + llama.

Ground truth was read out of preprocessing/out/rag_chunks_enriched.jsonl before the
run (not generated from the model). Each answerable question has `gold` (regex any
of which must appear in the answer) and `src` (substring of the source id that
should be among the cited sources). Unanswerable questions must be refused.
Scoring per question: retrieval hit (src cited), answer correct (gold in answer),
and for unanswerable: abstained. Results -> experiments/student_accuracy_coverage.jsonl
"""
import json, re, sys, time, urllib.request

URL = "http://127.0.0.1:7860/api/ask"
ABSTAIN = "could not find this"

Q = [
 # kind, question, gold regexes, source substring
 ("literal", "What is the minimum attendance required to write semester end exams in B.Tech R25?", [r"75\s*%"], "btech-r25-regulations"),
 ("literal", "What is the maximum number of credits for the B.Tech degree and how many must I earn to get the degree?", [r"164", r"160"], "btech-r25-regulations"),
 ("literal", "How many marks are there for CIE and SEE in a theory course under R25?", [r"40", r"60"], "btech-r25-regulations"),
 ("literal", "What grade point does an O grade carry?", [r"\b10\b"], "btech-r25-regulations"),
 ("literal", "What happens to a student whose attendance shortage is not condoned?", [r"detain"], "btech-r25-regulations"),
 ("literal", "How many credits is Design and Analysis of Algorithms in CSBS?", [r"\b3\b"], "csbs-r25"),
 ("literal", "What is the annual tuition fee for B.Tech Category-A students?", [r"1,59,600"], "admissions"),
 ("literal", "What is the highest package in the 2026 placement season?", [r"65\s*LPA|65 ?lakh"], "placements"),
 ("literal", "What NAAC grade does VNRVJIET have?", [r"A\+\+"], "about"),
 ("literal", "When was VNRVJIET established?", [r"1995"], "about"),
 ("literal", "How many volumes does the central library have?", [r"86,175"], "library"),
 ("literal", "How many seats does the Automobile Engineering department have?", [r"\b60\b"], "departments-ae"),
 # paraphrased
 ("paraphrase", "If my attendance is below the limit can I still sit for the final exams?", [r"condon|75|detain|not (be )?(permitted|eligible)"], "regulations"),
 ("paraphrase", "How is my CGPA worked out?", [r"credit|grade point|SGPA"], "btech-r25-regulations"),
 ("paraphrase", "What do I need to pass a course in terms of marks?", [r"40|C\b|pass"], "btech-r25-regulations"),
 ("paraphrase", "Which societies and clubs can I join on campus?", [r"NSS|Stentorian|Candleves|VNRSF|club"], "clubs"),
 ("paraphrase", "How do I pay my college fees online and which cards don't work?", [r"UPI|Internet|Rupay"], "fee-payment"),
 ("paraphrase", "What accreditation and ranking does the college have?", [r"NAAC|NIRF|NBA"], "about"),
 ("paraphrase", "Which clearances do I need before I leave the college with no dues?", [r"Library|Accounts|Hostel|Scholarship"], "no-dues"),
 ("paraphrase", "When do classes start for second year B.Tech students in 2026?", [r"06(th)? July|July 2026|07(th)? December"], "ii-year-academic-calendar"),
 # bridging (vocabulary differs)
 ("bridging", "Who do I complain to if seniors are ragging me?", [r"anti.?ragging|cell|committee"], "anti-ragging"),
 ("bridging", "Where can I study quietly and borrow books?", [r"library"], "library"),
 # unanswerable / private / out of scope -> must refuse
 ("unanswerable", "What are my internal marks this semester?", None, None),
 ("unanswerable", "What is the cafeteria menu for tomorrow?", None, None),
 ("unanswerable", "What is the phone number of my roll number's counsellor?", None, None),
 ("unanswerable", "Who won the IPL in 2020?", None, None),
]

def ask(q):
    body = json.dumps({"question": q, "q": q, "query": q, "role": "student", "mode": "coverage"}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

out = open("experiments/student_accuracy_coverage.jsonl", "w", encoding="utf-8")
rows = []
for i, (kind, q, gold, src) in enumerate(Q, 1):
    t = time.time()
    try:
        d = ask(q)
    except Exception as e:
        d = {"answer": "ERROR %s" % e, "citations": []}
    ans = d.get("answer", "")
    cites = " ".join(re.sub(r"[^a-z0-9]+", "-", (c.get("url") or "").lower()) for c in d.get("citations", []))
    abst = ABSTAIN in ans.lower()
    if gold is None:
        retr = None; correct = abst
    else:
        retr = bool(src and src.lower() in cites)
        correct = (not abst) and all(re.search(g, ans, re.I) for g in gold)
    rec = dict(n=i, kind=kind, q=q, abstained=abst, retrieval_hit=retr, correct=correct,
               secs=round(time.time() - t, 1), answer=ans[:400],
               cited=[c.get("url") for c in d.get("citations", [])][:5])
    rows.append(rec); out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    print("%2d %-12s %s retr=%s abst=%s  %s" % (i, kind, "OK " if correct else "BAD", retr, abst, q), flush=True)

print("\nBY KIND")
for k in ("literal", "paraphrase", "bridging", "unanswerable"):
    rs = [r for r in rows if r["kind"] == k]
    if rs:
        hit = [r for r in rs if r["retrieval_hit"] is not None]
        print("%-13s correct %d/%d%s" % (k, sum(r["correct"] for r in rs), len(rs),
              ("   retrieval-hit %d/%d" % (sum(r["retrieval_hit"] for r in hit), len(hit))) if hit else ""))
ans = [r for r in rows if r["kind"] != "unanswerable"]
print("ANSWERABLE accuracy %d/%d = %.0f%%" % (sum(r["correct"] for r in ans), len(ans), 100 * sum(r["correct"] for r in ans) / len(ans)))
print("OVERALL     accuracy %d/%d = %.0f%%" % (sum(r["correct"] for r in rows), len(rows), 100 * sum(r["correct"] for r in rows) / len(rows)))
