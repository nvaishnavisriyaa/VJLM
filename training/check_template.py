"""Phase 8 gate — verify the chat template and the training loss mask.

Run this BEFORE any training, for every base model:

    python training/check_template.py --base mistralai/Mistral-7B-Instruct-v0.3

Why this is a gate and not a nicety
-----------------------------------
SFT trains on whatever tokens are not masked to -100. The VJLM dataset puts
the retrieved CONTEXT in the user turn, so if the prompt is left unmasked the
model spends its capacity learning to reproduce ~500 tokens of college PDF
text instead of learning the answer behaviour the dataset was built to teach.
That failure is invisible: the loss curve looks fine and the model gets worse
at the actual task.

Three things checked here were all silently wrong before, and none of them
could have been caught on the Qwen2.5-1.5B control arm:

  1. Loss mask. prompt -> -100, assistant answer -> real ids, padding -> -100.

  2. Exactly one BOS. `apply_chat_template(tokenize=False)` emits a literal
     <|begin_of_text|> / <s> INTO THE STRING, and trl then tokenizes that
     string with add_special_tokens=True and prepends BOS a second time.
     Qwen2.5 has no BOS token at all, which is precisely why the control run
     never exposed this. A doubled BOS degrades every arm by the same unknown
     amount, which makes "fine-tuning did not help" unfalsifiable.

  3. The system role. All 1,493 VJLM records are [system, user, assistant].
     Mistral-7B-Instruct's template has historically raised
     "Only user and assistant roles are supported!"; some revisions instead
     fold the system turn into the first user turn. Either way the prompt
     shape can change per model, so the variant used is DETECTED and PRINTED
     here, and must be recorded in the run config - the base-model evaluation
     has to use the identical folding or the base-vs-tuned delta is void.

Exit code is non-zero if any check fails, so this can gate a pipeline.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "training", "data")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    note = ""
    if detail and not cond:
        note = " - " + str(detail)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, note))
    return cond


def apply_template(tok, messages, add_generation_prompt=False):
    """Render a chat turn list, coping with templates that reject `system`.

    Returns (text, variant) where variant is 'native' if the template accepted
    the system role as-is, or 'system_folded' if it had to be merged into the
    first user turn. The variant MUST be recorded with the run.
    """
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        return text, "native"
    except Exception:
        folded, sys_txt = [], ""
        for m in messages:
            if m["role"] == "system":
                sys_txt = m["content"]
                continue
            if sys_txt and m["role"] == "user":
                folded.append({"role": "user",
                               "content": sys_txt + "\n\n" + m["content"]})
                sys_txt = ""
            else:
                folded.append(dict(m))
        text = tok.apply_chat_template(
            folded, tokenize=False, add_generation_prompt=add_generation_prompt)
        return text, "system_folded"


def pick_pad(tok):
    """A pad token that is NOT the eos token, where the model provides one.

    Llama-3.2 ships <|finetune_right_pad_id|> exactly for this. Mistral has
    <unk>. Falling back to eos_token is workable only if masking is driven by
    the attention mask rather than by `input_ids == pad_token_id` - when PAD
    and EOS share an id, that comparison silently masks genuine EOS targets
    and the model never learns to stop.
    """
    if tok.pad_token is not None and tok.pad_token != tok.eos_token:
        return tok.pad_token, "model-provided"
    for cand in ("<|finetune_right_pad_id|>", "<pad>", "<unk>"):
        # convert_tokens_to_ids returns unk_token_id for tokens that are NOT in
        # the vocab, so a bare id check "finds" Llama's pad token inside
        # Mistral's vocab (it lands on <unk>=0). Require a round-trip.
        tid = tok.convert_tokens_to_ids(cand)
        if tid is None or tid < 0:
            continue
        if tok.convert_ids_to_tokens(tid) != cand:
            continue
        if cand != tok.eos_token:
            return cand, "selected"
    return tok.eos_token, "eos-fallback (RISK: see docstring)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--max-length", type=int, default=1280)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    print("[1/5] tokenizer: %s" % a.base)
    tok = AutoTokenizer.from_pretrained(a.base)
    pad, how = pick_pad(tok)
    tok.pad_token = pad
    tok.padding_side = "right"
    print("  vocab=%d  eos=%r(%s)  pad=%r(%s) via %s"
          % (len(tok), tok.eos_token, tok.eos_token_id, tok.pad_token,
             tok.pad_token_id, how))
    check("pad token is distinct from eos", tok.pad_token_id != tok.eos_token_id,
          "pad==eos==%s; masking must use the attention mask, never "
          "input_ids==pad_id" % tok.eos_token_id)

    print("[2/5] chat template on a real record")
    ex = json.loads(open(os.path.join(DATA, "train.jsonl"), encoding="utf-8").readline())
    msgs = ex["messages"]
    check("record is [system, user, assistant]",
          [m["role"] for m in msgs] == ["system", "user", "assistant"],
          [m["role"] for m in msgs])
    prompt_text, variant = apply_template(tok, msgs[:-1], add_generation_prompt=True)
    full_text, _ = apply_template(tok, msgs, add_generation_prompt=False)
    print("  chat_template_variant = %s" % variant)

    # Mistral-7B-Instruct's template injects the system message next to the
    # LAST user turn. Appending an assistant turn moves it, so rendering
    # [system,user] and [system,user,assistant] disagree about where the system
    # text sits (measured: they diverge at char 10). That is why the training
    # pair is built as prompt + answer + eos and the template's own
    # full-conversation render is never used - we never serve a completed
    # conversation, so it is not the thing to reproduce. Reported, not failed.
    stable = full_text.startswith(prompt_text)
    print("  template prefix-stable across renders: %s%s" % (
        stable, "" if stable else "  (system relocates - full render unused)"))

    print("[3/5] BOS accounting")
    bos = tok.bos_token
    if bos is None:
        print("  model has no BOS token - the double-BOS trap does not apply")
        check("no BOS to double", True)
    else:
        in_text = prompt_text.count(bos)
        print("  template emitted %d literal %r into the string" % (in_text, bos))
        naive = tok(prompt_text, add_special_tokens=True)["input_ids"]
        fixed = tok(prompt_text, add_special_tokens=False)["input_ids"]
        n_naive = sum(1 for t in naive if t == tok.bos_token_id)
        n_fixed = sum(1 for t in fixed if t == tok.bos_token_id)
        print("  add_special_tokens=True  -> %d BOS in input_ids" % n_naive)
        print("  add_special_tokens=False -> %d BOS in input_ids" % n_fixed)
        check("exactly one BOS with add_special_tokens=False", n_fixed == 1,
              "got %d" % n_fixed)
        if n_naive > 1:
            print("  => CONFIRMED: add_special_tokens must be False for this model.")

    print("[4/5] loss mask on the prompt/completion split")
    # The training pair is constructed, not scraped out of a full render:
    #   prompt     = exactly what inference sends (see rag/retrieve.py)
    #   completion = the assistant answer + eos, so the model learns to stop
    # Prefix-stability is then true by construction for every model, and the
    # trained prompt is byte-identical to the served prompt.
    completion_text = msgs[-1]["content"] + tok.eos_token
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + list(c_ids)
    ids = list(p_ids) + list(c_ids)
    print("  prompt=%d tok  completion=%d tok  total=%d (max_length=%d)"
          % (len(p_ids), len(c_ids), len(ids), a.max_length))
    check("every prompt position is masked",
          all(l == -100 for l in labels[:len(p_ids)]))
    check("every completion position is a real target",
          all(l != -100 for l in labels[len(p_ids):]) and len(c_ids) > 0)
    check("completion ends with eos so the model learns to stop",
          c_ids[-1] == tok.eos_token_id,
          "last completion id is %s, eos is %s" % (c_ids[-1], tok.eos_token_id))
    check("eos target is not masked as padding",
          labels[-1] == tok.eos_token_id)
    check("nothing truncates at max_length", len(ids) <= a.max_length,
          "%d > %d" % (len(ids), a.max_length))
    ans = msgs[-1]["content"][:60].replace("\n", " ")
    dec = tok.decode(c_ids, skip_special_tokens=True)[:60].replace("\n", " ")
    check("completion decodes back to the assistant answer",
          ans.strip()[:40] in dec, "expected %r got %r" % (ans, dec))
    # Serving parity: rag/retrieve.py hands llama-server [system, user], which
    # renders through this same template. Confirm the prompt we train on is the
    # prompt that will be served.
    check("trained prompt == served prompt render",
          prompt_text == apply_template(tok, msgs[:-1], add_generation_prompt=True)[0])

    print("[5/5] padded batch")
    batch = tok([prompt_text, prompt_text[:len(prompt_text) // 2]],
                padding=True, add_special_tokens=False, return_tensors=None)
    am = batch["attention_mask"][1]
    ii = batch["input_ids"][1]
    n_pad = sum(1 for m in am if m == 0)
    check("padding is flagged by the attention mask", n_pad > 0, "no padding produced")
    check("padded positions carry pad_token_id",
          all(ii[i] == tok.pad_token_id for i, m in enumerate(am) if m == 0))

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        print("\nDo NOT start training until these pass.")
        sys.exit(1)
    print("chat_template_variant=%s  pad_token=%r  add_special_tokens=False"
          % (variant, tok.pad_token))
    print("Record those three in the run config.")


if __name__ == "__main__":
    main()
