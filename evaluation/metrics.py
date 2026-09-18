"""Scoring for the VJLM behavioural test set - ONE implementation.

Extracted from eval_generation.py when a second generation backend was added.
Two copies of a metric drift, and when they do you cannot tell whether a
difference between two runs came from the models or from the scorer. Every
backend imports from here.

What these metrics do and do not measure
----------------------------------------
`grounded_f1` is unigram token-SET F1 against the reference answer. It is a
weak proxy in a specific, directional way: it rewards overlap, so a model that
parrots the retrieved context back can outscore a correctly concise one - and
concision is plausibly part of what this dataset teaches. `mean_answer_tokens`
is therefore reported alongside it, and a gain that disappears once answer
length is matched is not a gain.

Abstention is detected by substring markers, which is why the prompt in
rag/retrieve.py pins one exact refusal sentence. A model that refuses in its
own words scores as a false answer; that is a known limitation of the metric,
not of the model.

`bootstrap_ci` exists because the test set is small and lopsided: 60 of 147
items are abstention cases, so grounded_f1 is computed over only 87 items.
Differences of a few points on 87 items are noise, and the matrix makes
several comparisons at once. Per-item scores are always written out so a CI,
a paired test, or a re-score with a better metric never needs the 5 x 147
generations to be repeated.
"""
import random, re

ABSTAIN_MARKERS = ["could not find", "not find this", "no information",
                   "not available in", "does not contain"]
CITE = re.compile(r"(Source:\s*http|\[\d\])")
WORD = re.compile(r"[a-z0-9]+")


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


def score_item(pred, ref, n_pred_tokens=None):
    """Per-item record. Written to disk so aggregates can be recomputed."""
    ref_abstain = is_abstain(ref)
    pred_abstain = is_abstain(pred)
    return {
        "ref_abstain": ref_abstain,
        "pred_abstain": pred_abstain,
        "f1": None if ref_abstain else f1(pred, ref),
        "cited": bool(CITE.search(pred)),
        "pred_chars": len(pred),
        "pred_tokens": n_pred_tokens,
        "pred": pred,
    }


def bootstrap_ci(values, n=2000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI for a mean. Returns (lo, hi) or None."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    k = len(vals)
    means = []
    for _ in range(n):
        means.append(sum(vals[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return [round(lo, 4), round(hi, 4)]


def aggregate(items, n_requested, n_failed, seed=0):
    """Roll per-item records up into one result row.

    Divides by the number of items actually SCORED, never by the number
    requested. The original implementation counted failed requests in the
    denominator, which deflated citation_rate and sec_per_answer in proportion
    to the failure rate - and since the 7B arm is the slowest and most
    timeout-prone, that bias was directional and landed on exactly the arm
    under test. 20 dropped requests out of 147 deflate citation_rate by 14%,
    which is more than enough to reorder the arms.
    """
    n_ok = len(items)
    f1s = [i["f1"] for i in items if i["f1"] is not None]
    exp = [i for i in items if i["ref_abstain"]]
    got = [i for i in items if i["pred_abstain"]]
    correct = [i for i in items if i["ref_abstain"] and i["pred_abstain"]]
    false_ab = [i for i in items if not i["ref_abstain"] and i["pred_abstain"]]
    tok_counts = [i["pred_tokens"] for i in items if i.get("pred_tokens")]
    return {
        "n_requested": n_requested,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_f1_items": len(f1s),
        "n_abstain_items": len(exp),
        "grounded_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "grounded_f1_ci95": bootstrap_ci(f1s, seed=seed),
        "abstain_recall": round(len(correct) / len(exp), 4) if exp else None,
        "abstain_precision": round(len(correct) / len(got), 4) if got else None,
        "false_abstain": len(false_ab),
        "citation_rate": round(sum(1 for i in items if i["cited"]) / n_ok, 4) if n_ok else 0.0,
        "mean_answer_chars": round(sum(i["pred_chars"] for i in items) / n_ok) if n_ok else 0,
        "mean_answer_tokens": round(sum(tok_counts) / len(tok_counts)) if tok_counts else None,
    }
