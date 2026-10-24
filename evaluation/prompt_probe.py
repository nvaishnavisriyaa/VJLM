"""Does a different prompt make the model's OWN answer better?

Retrieval is held fixed: each question is retrieved once and the same context is
sent under each prompt variant, so any difference is the prompt. Only questions
whose answer is in the prompt text are used - for the rest no prompt can help.

Scores the RAW model output (no rag/answer.py post-processing), at temperature 0
with a fixed seed. The model is fine-tuned on one prompt shape ("CONTEXT: ...
QUESTION: ..."), so every variant keeps that shape and only adds instructions.

    python evaluation/prompt_probe.py [n_questions] [variants,comma,separated]

Needs llama-server on :8080. Writes experiments/prompt_probe.jsonl.
"""
import json, os, re, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
os.chdir(ROOT)
from student_questions import Q as Q1          # noqa: E402
from student_questions_v2 import Q2            # noqa: E402
from rag import retrieve as R                  # noqa: E402

ABSTAIN = "could not find this"
sys.stdout.reconfigure(encoding="utf-8")

DIRECT = ("Reply with the direct answer first, in one to three sentences. Give the exact "
          "numbers, limits, dates and conditions the CONTEXT states for this question, "
          "and leave out clauses the question did not ask about.")


def v_current(msgs):
    return msgs


def v_system(msgs):
    """Same shape; the instruction lives in the system message."""
    m = [dict(x) for x in msgs]
    m[0]["content"] = m[0]["content"].replace(
        "Never invent dates, names, fees or policies.",
        "Never invent dates, names, fees or policies. " + DIRECT)
    return m


def v_tail(msgs):
    """Same system message; the instruction follows the question."""
    m = [dict(x) for x in msgs]
    m[1]["content"] = m[1]["content"] + "\n\n" + DIRECT + " Cite sources as [1], [2]."
    return m


TAIL2 = ("Answer in one or two plain sentences, giving the exact figures, limits and "
         "conditions from the CONTEXT. Do not copy clause numbers. Cite sources as [1], [2]. "
         "If the CONTEXT does not contain the answer, say: 'I could not find this in the "
         "available VNRVJIET sources.'")


def v_tail2(msgs):
    m = [dict(x) for x in msgs]
    m[1]["content"] = m[1]["content"] + "\n\n" + TAIL2
    return m


VARIANTS = {"current": v_current, "system": v_system, "tail": v_tail, "tail2": v_tail2}

# Must stay refused: not in any public document, or a request for the user's own record.
REFUSE = ["What are my internal marks?", "Who won the IPL in 2020?",
          "What is the share price of Infosys today?", "What is my CGPA?",
          "What is the Wi-Fi password of the boys hostel?", "What did the cafeteria serve today?"]


def llama(messages):
    body = json.dumps({"messages": messages, "max_tokens": 420, "temperature": 0,
                       "seed": 7, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def gold_in_prompt(body, allx, src):
    ctx = body.split("CONTEXT:\n", 1)[-1]
    for p in re.split(r"\n\n(?=\[\d+\] )", ctx):
        if src and not re.search(src, p.split("\n", 1)[0], re.I):
            continue
        if all(re.search(a, p, re.I) for a in allx):
            return True
    return False


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    names = (sys.argv[2] if len(sys.argv) > 2 else "current,tail,tail2").split(",")
    r = R.Retriever()
    qs = Q1 + Q2
    picked = []
    for qid, split, kind, q, allx, src in qs:
        res = r.search(q, role="student", topk=5, mode="fusion")
        msgs = R.build_prompt(q, res, role="student")
        if gold_in_prompt(msgs[1]["content"], allx, src):
            picked.append((qid, q, allx, msgs))
    # every k-th so both question sets and all kinds are represented
    step = max(1, len(picked) // n)
    picked = picked[::step][:n]
    print("questions with the answer in the prompt: %d (using %d)" % (len(picked), len(picked)), flush=True)

    out = open(os.path.join(ROOT, "experiments", "prompt_probe.jsonl"), "w", encoding="utf-8")
    tot = {v: dict(strict=0, abst=0, words=0) for v in names}
    for i, (qid, q, allx, msgs) in enumerate(picked, 1):
        for v in names:
            t0 = time.time()
            ans = llama(VARIANTS[v](msgs))
            strict = all(re.search(a, ans, re.I) for a in allx)
            ab = ABSTAIN in ans.lower()
            tot[v]["strict"] += strict
            tot[v]["abst"] += ab
            tot[v]["words"] += len(ans.split())
            out.write(json.dumps({"qid": qid, "variant": v, "q": q, "strict": strict,
                                  "abstained": ab, "secs": round(time.time() - t0, 1),
                                  "answer": ans}) + "\n")
            out.flush()
        print("%2d/%d %-4s %s" % (i, len(picked), qid, "  ".join(
            "%s=%d/%d" % (v, tot[v]["strict"], i) for v in names)), flush=True)
    print("\nvariant   strict  abstained  avg words")
    for v in names:
        t = tot[v]
        print("%-8s  %2d/%d   %2d         %.0f" % (v, t["strict"], len(picked), t["abst"],
                                                  t["words"] / len(picked)))
    refusals(names, r)


def refusals(names, r):
    print("\nQuestions that must be REFUSED (abstained / total):", flush=True)
    rows = []
    for q in REFUSE:
        res = r.search(q, role="student", topk=5, mode="fusion")
        msgs = R.build_prompt(q, res, role="student")
        for v in names:
            ans = llama(VARIANTS[v](msgs))
            rows.append((v, q, ABSTAIN in ans.lower(), ans))
    for v in names:
        k = [x for x in rows if x[0] == v]
        print("  %-8s refused %d/%d" % (v, sum(x[2] for x in k), len(k)), flush=True)
        for _, q, ab, ans in k:
            if not ab:
                print("      NOT refused: %s -> %s" % (q, ans[:140].replace("\n", " ")), flush=True)


if __name__ == "__main__":
    main()