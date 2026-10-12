"""Student question set for retrieval measurement (RAG-EVAL-002).

Each answerable question carries `all`: regexes that must ALL match inside ONE
retrieved chunk for it to count as a hit (the chunk really contains the fact),
and optionally `src`: a regex the chunk URL must match (right document family).
Gold was read out of the corpus, never produced by the model. validate() fails
loudly if a gold matches nothing, so an unanswerable "answerable" question
cannot sit in the set silently.

Split: "dev" is used while tuning; "test" is read only for the final number.
"""
# (id, split, kind, question, all-regexes, src-regex)
Q = [
 # ---- R25 B.Tech regulations (literal)
 ("r01","dev","literal","What is the minimum attendance required to write semester end exams in B.Tech R25?",[r"75\s*%",r"aggregate attendance"],r"btech-r25"),
 ("r02","test","literal","How many credits must a B.Tech student earn to be awarded the degree under R25?",[r"164",r"160"],r"btech-r25"),
 ("r03","dev","literal","How many marks does CIE carry and how many does SEE carry in R25 B.Tech?",[r"Continuous Internal Evaluation \(CIE\): 40 Marks",r"Semester End Examination \(SEE\): 60"],r"btech-r25"),
 ("r04","test","literal","What grade point is given for the O grade in B.Tech R25?",[r"Outstanding\s*10"],r"btech-r25"),
 ("r05","dev","literal","What happens to a B.Tech R25 student whose shortage of attendance is not condoned?",[r"not condoned shall be detained"],r"btech-r25"),
 ("r06","test","literal","What is the minimum percentage of marks to get a pass grade C in B.Tech R25?",[r"C\s*Pass\s*5"],r"btech-r25"),
 ("r07","dev","literal","What does SGPA stand for in the B.Tech regulations?",[r"SGPA\s+Semester Grade Point Average"],r"btech-r25"),
 ("r08","test","literal","What is the full form of CoE in the academic regulations?",[r"CoE\s+Controller of Examinations"],r"btech-r25"),
 # ---- R25 paraphrased
 ("r09","dev","paraphrase","If my attendance is below the limit can I still sit for the final exams?",[r"condon"],r"btech-r25"),
 ("r10","test","paraphrase","How is my semester grade point average calculated?",[r"SGPA"],r"btech-r25"),
 ("r11","dev","paraphrase","What marks do I need to clear a subject?",[r"Minimum Requirement for Pass|Fail"],r"btech-r25"),
 ("r12","test","paraphrase","Can I get extra attendance for writing the mid exam?",[r"Attendance credit equivalent to two lecture periods"],r"btech-r25"),
 ("r13","dev","paraphrase","What if I miss the end semester exam, is there a repeat exam?",[r"Supplementary Student|Supplementary Examinations"],r"btech-r25"),
 ("r14","test","paraphrase","Can I challenge my marks after results?",[r"Revaluation"],r"btech-r25"),
 # ---- explicit older regulation (must NOT be forced to R25)
 ("o01","dev","literal","What is the attendance requirement in the R22 B.Tech regulations?",[r"75\s*%",r"attendance"],r"r22"),
 ("o02","test","literal","What are the credits required for the award of degree in R19?",[r"credits"],r"vnrr19"),
 # ---- M.Tech / MCA (must still reach their own documents)
 ("m01","dev","literal","What is the minimum attendance for M.Tech R25 students?",[r"75\s*%",r"attendance"],r"vnrmtr25"),
 ("m02","test","literal","What is the attendance eligibility for MCA R25 semester end examinations?",[r"75\s*%",r"attendance"],r"mca-r25"),
 # ---- course / scheme
 ("c01","dev","literal","How many credits is Design and Analysis of Algorithms in CSBS second year?",[r"25PC1CB203 Design and Analysis of Algorithms 3 0 0 3 3"],r"csbs-r25"),
 ("c02","test","literal","What is the credit structure of Operating Systems in CSBS R25?",[r"25PC1CB206 Operating Systems 3 0 0 3 3"],r"csbs-r25"),
 # ---- fees / admissions
 ("f01","dev","literal","What is the annual tuition fee for B.Tech Category-A students?",[r"1,59,600"],r"admissions"),
 ("f02","test","literal","How much is the M.Tech annual fee?",[r"1,30,000"],r"admissions"),
 ("f03","dev","paraphrase","How do I pay my college fees online and which cards are not allowed?",[r"Rupay"],r"fee-payment"),
 ("f04","test","paraphrase","Which payment methods can I use for tuition and hostel fee?",[r"UPI",r"Internet Banking"],r"fee-payment"),
 ("f05","dev","literal","What documents do I need to attend spot admissions?",[r"original certificates"],r"spot"),
 # ---- placements / about
 ("p01","dev","literal","What is the highest package in the 2026 placement season?",[r"65 LPA"],r"placements"),
 ("p02","test","literal","How many total offers did the 2026 batch get?",[r"1,974\s+Total\s+Offers"],r"placements"),
 ("p03","dev","paraphrase","What is the average salary package for placed students?",[r"Average CTC"],r"placements"),
 ("a01","test","literal","What NAAC grade does VNRVJIET have?",[r"A\+\+"],r"/about"),
 ("a02","dev","literal","When was VNRVJIET established?",[r"In 1995"],r"/about"),
 ("a03","test","paraphrase","What accreditation and ranking does the college have?",[r"NAAC",r"NIRF"],r"/about"),
 ("a04","dev","literal","Which university is VNRVJIET affiliated to?",[r"affiliated to Jawaharlal Nehru Technological University"],r"/about"),
 # ---- campus
 ("l01","dev","literal","How many volumes does the central library have?",[r"86,175"],r"library"),
 ("l02","test","paraphrase","Where can I read online journals and e-resources?",[r"IEEE Xplore"],r"library"),
 ("l03","dev","literal","How many computers does the digital library have and until what time is it open?",[r"20 computers",r"8 p\.m\."],r"library"),
 ("k01","test","paraphrase","Which societies and clubs can I join on campus?",[r"NSS",r"Stentorian"],r"clubs"),
 ("k02","dev","literal","What is Stentorian club about?",[r"Stentorian"],r"clubs"),
 # ---- grievance / forms
 ("g01","test","bridging","Who do I complain to if seniors are ragging me?",[r"[Rr]agging"],r"anti-ragging"),
 ("g02","dev","bridging","Where do I report sexual harassment on campus?",[r"Internal Complaints|ICC"],r"icc|internal-complaints"),
 ("g03","test","paraphrase","Which sections need to give clearance before I get my no dues certificate?",[r"Library",r"Accounts Section"],r"dues"),
 ("g04","dev","literal","Where can I download the B.Tech condonation form?",[r"B\.Tech Condonation Form"],r"academics"),
 # ---- calendar
 ("d01","dev","literal","When do classes commence for II year B.Tech in 2026-27?",[r"06(th)? July 2026"],r"ii-year-academic-calendar|class-commencement"),
 ("d02","test","literal","When does the first year B.Tech academic year 2026-27 start classes?",[r"Commencement of Classes \| 10 August 2026"],r"i-year-academic-calendar"),
 # ---- departments
 ("t01","dev","literal","How many seats does the Automobile Engineering department have?",[r"Automobile",r"60\s+Seats\s+Intake"],r"departments"),
 ("t02","test","literal","When was the Automobile Engineering department established?",[r"Established in 2010"],r"departments"),

 # ---- HOLDOUT (written after tuning; never used to tune; run once for the final number)
 ("h01","hold","literal","What is the anti-ragging helpline number?",[r"1800-180-5522"],r"anti-ragging"),
 ("h02","hold","literal","When was the Research and Development Cell established?",[r"Established 2006"],r"/research"),
 ("h03","hold","literal","How many buses are in the college transport fleet?",[r"40\+ buses"],r"campus-life"),
 ("h04","hold","paraphrase","Which gateway do I use to pay tuition fees online?",[r"Axis Bank Easypay"],r"payment"),
 ("h05","hold","paraphrase","Who handles the conduct of exams and result processing?",[r"result-processing"],r"examination-cell"),
 ("h06","hold","literal","Under which AICTE regulation was the Grievance Redressal Committee formed?",[r"AICTE regulations, 2012"],r"grievance"),
 ("h07","hold","literal","Where can I download the holiday list for 2026?",[r"Holidays 2026"],r"academics"),
 ("h08","hold","literal","What is NSS and which ministry runs it?",[r"Ministry of Youth Affairs"],r"clubs/nss"),
 ("h09","hold","paraphrase","Is there an honour code or code of conduct I must follow as a student?",[r"Academic Honour Code"],r"Code-of-Ethics"),
 ("h10","hold","literal","What sports facilities does the campus have?",[r"Sports Complex"],r"campus-life"),
 ("h11","hold","literal","Where do I download the M.Tech no dues form?",[r"M\.Tech No Dues Form"],r"academics"),
 ("h12","hold","literal","How many marks is the semester end exam in R22 B.Tech?",[r"SEE - 60 M"],r"r22"),
 ("h13","hold","paraphrase","Are there canteens or food courts on campus?",[r"Food Courts"],r"campus-life"),
 ("h14","hold","paraphrase","Where can I see who is on the anti-ragging committee?",[r"Anti-Ragging Committees"],r"anti-ragging"),
 ("h15","hold","paraphrase","Where do I pay my exam fee online?",[r"Examination Fee"],r"payment"),
 ("h16","hold","literal","Which branches have vacant seats in spot admissions for B.Tech 2026?",[r"Automobile, Biotechnology"],r"spot"),
 ("h17","hold","literal","Who is the Principal of the college?",[r"Chennakesava Rao\s+Principal"],None),
]
UNANSWERABLE = [
 "What are my internal marks this semester?",
 "What is the cafeteria menu for tomorrow?",
 "Who won the IPL in 2020?",
 "What is my roll number's attendance percentage today?",
]

def validate():
    import json, re
    rows = [json.loads(l) for l in open("preprocessing/out/rag_chunks_enriched.jsonl", encoding="utf-8")]
    bad = []
    for qid, split, kind, q, allx, src in Q:
        pats = [re.compile(a, re.I) for a in allx]
        s = re.compile(src, re.I) if src else None
        n = sum(1 for r in rows if all(p.search(r["text"]) for p in pats)
                and (s is None or s.search(r["url"])))
        if n == 0:
            bad.append((qid, q))
    return bad

if __name__ == "__main__":
    b = validate()
    print(len(Q), "questions;", len(b), "have no matching chunk in the corpus")
    for x in b: print("  NO GOLD:", x)
