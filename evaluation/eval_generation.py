"""Score a served model on the held-out VJLM behavioural test set.

Run once against the BASE model and once against the FINE-TUNED model, with
everything else identical. The difference is the only evidence that fine-tuning
helped. Without the base number, "the fine-tune worked" is unfalsifiable.

Usage:
    # start llama-server on the base build, then:
    python evaluation/eval_generation.py --label base
    # restart llama-server on the vjlm build, then:
    python evaluation/eval_generation.py --label finetuned
    python evaluation/eval_generation.py --compare
"""
import argparse, json, os, re, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "training", "data", "test.jsonl")
OUT = os.path.join(ROOT, "experiments", "generation_results.jsonl")
URL = os.environ.get("VJLM_LLAMA_URL", "http://127.0.0.1:8080")

ABSTAIN_MARKERS = ["could not find", "not find this", "no information",
                   "not available in", "does not contain"]
CITE = re.compile(r"(Source:\s*http|\[\d\])")
WORD = re.compile(r"[a-z0-9]+")


def gen(messages, max_tokens=320):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def toks(s):
    return set(WORD.findall(s.lower()))


def f1(pred, ref):
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return 0.0
    tp = len(p & r)
    if not tp:
        return 0.0
    prec, rec = tp / len(p), tp / len(r)
    return 2 * prec * rec / (prec + rec)


def is_abstain(t):
    lt = t.lower()
    return any(m in lt for m in ABSTAIN_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.compare:
        rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
        by = {}
        for r in rows:
            by[r["label"]] = r
        keys = ["n", "grounded_f1", "abstain_recall", "abstain_precision",
                "citation_rate", "false_abstain", "sec_per_answer"]
        print("%-22s %s" % ("metric", "  ".join("%-12s" % k for k in by)))
        for k in keys:
            print("%-22s %s" % (k, "  ".join("%-12s" % by[l].get(k, "-") for l in by)))
        return

    data = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    if a.limit:
        data = data[:a.limit]

    n_abst_expected = n_abst_got = n_abst_correct = 0
    n_false_abstain = 0
    f1s, cites = [], 0
    t0 = time.time()

    for i, ex in enumerate(data):
        msgs = ex["messages"]
        ref = msgs[-1]["content"]
        prompt = msgs[:-1]
        try:
            pred = gen(prompt)
        except Exception as e:
            print("  request failed:", str(e)[:90])
            continue

        ref_abstain = is_abstain(ref)
        pred_abstain = is_abstain(pred)
        if ref_abstain:
            n_abst_expected += 1
            if pred_abstain:
                n_abst_correct += 1
        else:
            if pred_abstain:
                n_false_abstain += 1
        if pred_abstain:
            n_abst_got += 1

        if not ref_abstain:
            f1s.append(f1(pred, ref))
        if CITE.search(pred):
            cites += 1

        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, len(data)), flush=True)

    n = len(data)
    dur = time.time() - t0
    res = {
        "label": a.label, "n": n,
        "grounded_f1": round(sum(f1s) / len(f1s), 3) if f1s else 0.0,
        "abstain_recall": round(n_abst_correct / n_abst_expected, 3) if n_abst_expected else None,
        "abstain_precision": round(n_abst_correct / n_abst_got, 3) if n_abst_got else None,
        "false_abstain": n_false_abstain,
        "citation_rate": round(cites / n, 3),
        "sec_per_answer": round(dur / n, 1),
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    print("\n" + json.dumps(res, indent=2))
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    print("appended ->", OUT)


if __name__ == "__main__":
    main()
