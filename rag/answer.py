"""Answer post-processing: citations written by code, not by the model (APP-002).

Why this exists
---------------
The fine-tuned model ends every answer with "Source: <url> (page n)" - free
text it chooses itself. In the first student test it cited the /payment page
for facts (Rupay not allowed, UPI) that only the fee-guidelines PDF contains.
A reader who opened the cited page could not find the claim: a citation that
looks like evidence and is not is worse than no citation.

So the model's own Source line is discarded and replaced with attribution
computed from the text: each answer sentence is matched to the retrieved
passage that contains its content words, and the inline [n] and the final
"Sources" line come from that match. A sentence no passage supports gets no
citation and is reported as unsupported, which the UI can show.

Also removed here: two boilerplate sentences the model learned from the
role-differentiation training rows (training/build_dataset.py) and appends to
every student answer - "In simple terms:" and "check this against your own
regulation batch, since the rule can differ between R18, R19, ...". The
second is not in the retrieved context, so it is an unsupported claim.
"""
import re

from rag.index import tok

_SOURCE_LINE = re.compile(r"\n*\s*(?:Sources?|Source URL)\s*:.*$", re.I | re.S)
_HEDGE = re.compile(
    r"(?:what this means for you as a student\s*:\s*)?"
    r"(?:please\s+)?check this against your own regulation batch[^.\n]*[.\n]?"
    r"(?:\s*since the (?:rule|attendance requirements?|provisions?)[^.\n]*[.\n]?)?",
    re.I)
_HEDGE2 = re.compile(r"[^.\n]*\b(?:differ|differs|vary|varies)\b[^.\n]*\bR18\b[^.\n]*\.?", re.I)
# The same trained-in hedge in its other wordings - "confirm which regulation
# batch applies before applying this, as the provisions differ across
# regulation versions", which once came back as an entire answer. Keyed on
# "regulation batch"/"regulation version", which is the training template's
# phrasing; the regulations themselves say "R22 Regulations", never this.
_HEDGE3 = re.compile(r"[^.\n]*\bregulation (?:batch|version)s?\b[^.\n]*\.?", re.I)
_LEAD = re.compile(r"^\s*in simple terms\s*:\s*", re.I)
_ABSTAIN = "could not find this"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9••\-])|\n+")

SUPPORT_MIN = 0.55      # share of a sentence's content words found in a passage


def clean(text):
    """Strip the model's own Source line and the learned boilerplate."""
    t = _SOURCE_LINE.sub("", text or "")
    t = _LEAD.sub("", t)
    t = _HEDGE.sub("", t)
    t = _HEDGE2.sub("", t)
    t = _HEDGE3.sub("", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _forms(w):
    out = {w}
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        out.add(w[:-2])
    if w.endswith("s") and len(w) > 3:
        out.add(w[:-1])
    else:
        out.add(w + "s")
    return out


def _support(sentence, passage_tokens):
    """Share of the sentence's content words (incl. numbers) the passage has."""
    toks = [t for t in tok(sentence) if len(t) > 2 or t.isdigit()]
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if _forms(t) & passage_tokens)
    return hit / len(toks)


def _name(c):
    n = (c.get("url") or "").rstrip("/").split("/")[-1] or c.get("url") or "source"
    return n


def attribute(text, chunks):
    """-> (final_text, used_indices, unsupported_sentences).

    final_text carries an inline [n] after each supported sentence and a
    'Sources:' line listing exactly the passages that supported something.
    """
    body = clean(text)
    if not body or _ABSTAIN in body.lower() or not chunks:
        return body, [], []
    # PDF text arrives hard-wrapped; a wrapped line is not a sentence.
    body = re.sub(r"(?<!\n)\n(?!\n|[•*-]|\d+[.)]\s)", " ", body)
    ptoks = [set(tok(c["text"])) for c in chunks]

    out_parts, used, unsupported = [], [], []
    for line in body.split("\n"):
        if not line.strip():
            out_parts.append(line)
            continue
        pieces = [p for p in _SENT.split(line) if p and p.strip()]
        rebuilt = []
        for s in pieces:
            best, best_i = 0.0, None
            for i, pt in enumerate(ptoks):
                sc = _support(s, pt)
                if sc > best + 1e-9:
                    best, best_i = sc, i
            if best_i is not None and best >= SUPPORT_MIN:
                if best_i not in used:
                    used.append(best_i)
                s = s.rstrip() + " [%d]" % (best_i + 1)
            else:
                unsupported.append(s.strip())
            rebuilt.append(s)
        out_parts.append(" ".join(rebuilt))
    final = "\n".join(out_parts).strip()
    if used:
        used_sorted = sorted(used)
        final += "\n\nSources: " + "; ".join(
            "[%d] %s (page %s)" % (i + 1, _name(chunks[i]), chunks[i].get("page"))
            for i in used_sorted)
    return final, sorted(used), unsupported


def with_supplement(query, final, chunks, idf):
    """Append the other relevant sentences from the retrieved passages that the
    answer did not say (rag/focus.py). Each carries its own [n]."""
    from rag.focus import supplement
    if not final or _ABSTAIN in final.lower() or not chunks:
        return final
    body = final.split(chr(10) + chr(10) + "Sources:")[0]
    extra = supplement(query, chunks, body, idf, used=used_from(final, len(chunks)))
    if not extra:
        return final
    lines = ["- %s [%d]" % (u.rstrip(), ci + 1) for ci, u in extra]
    head, sep, tail = final.partition(chr(10) + chr(10) + "Sources:")
    used = sorted({ci for ci, _ in extra} | {int(n) - 1 for n in re.findall(r"\[(\d+)\]", head)})
    srcs = "; ".join("[%d] %s (page %s)" % (i + 1, _name(chunks[i]), chunks[i].get("page")) for i in used if i < len(chunks))
    return (head + chr(10) + chr(10) + "Also stated in the sources:" + chr(10) + chr(10).join(lines)
            + chr(10) + chr(10) + "Sources: " + srcs)


# A request for the value of the reader's own record - "what are my internal
# marks", "the phone number of my roll number's counsellor". The corpus
# describes rules, never a student's records, so these must stay refused.
# "if my attendance is below 75% can I sit the exam" is NOT one of these: it
# asks what the rule says, and the wh-word test separates the two.
_PERSONAL = re.compile(r"\b(what|which|who|when|how much|how many)\b[^?]{0,40}\bmy\b", re.I)


def _topic_present(query, text, idf):
    """Is the question's most discriminating word actually in this passage?

    The retrieval score cannot answer this: measured, "who won the IPL in
    2020" scores 0.788 against the corpus and the perfectly answerable "if my
    attendance is below the limit can I still sit for the final exams" scores
    0.787. What separates them is that `attendance` is in the retrieved
    passage and `ipl` and `cafeteria` are nowhere in it. One rare term
    present is a far better answerability test than any score threshold.
    """
    from rag.scope import content_terms, variants
    # Only terms the index knows can be ranked by rarity. A word the index
    # has never seen ("my") would otherwise look maximally rare and be picked
    # as the topic, blocking every rescue.
    ts = [t for t in content_terms(tok(query or "")) if t in idf]
    if not ts:
        return False
    from rag.synonyms import DOMAIN
    top = sorted(ts, key=lambda t: -idf[t])[:3]
    toks = set(tok(text))
    return any(t in toks or (variants(t) & toks) or (set(DOMAIN.get(t, ())) & toks)
               for t in top)


def rescue(query, chunks, idf, min_score=0.15, max_units=3):
    """-> extractive answer when the model refused but a passage plainly answers.

    The model was trained on one-passage contexts whose target is the first
    sentence of that passage, so when a question's wording differs from the
    clause's it answers "I could not find this" with the clause sitting at
    rank 1 (measured: "if my attendance is below the limit can I still sit for
    the final exams?" against the R25 attendance clause). Refusing when the
    answer was retrieved is the worst of both failures, so the passage's own
    sentences are returned instead.

    Only the top-ranked passage is used - coverage already judged it the best
    answer to this question, and second-guessing that with a second scorer is
    how the first attempt failed. Unit scores only choose WHICH sentences of
    it to quote: the right clause here ("attendance below 65% shall not be
    condoned") scores just 0.23 against the question's words, because the
    rule says `condoned` where the student says `sit the exam`, so the bar
    for a sentence is deliberately low and the guards carry the safety.

    Three guards keep genuine refusals refused: requests for the reader's own
    records are never rescued, the question's rarest word must appear in the
    passage, and every identifier in the question must appear in it too.
    """
    from rag.focus import scored_units, _prose
    from rag.scope import identifiers
    if not chunks or _PERSONAL.search(query or ""):
        return None
    c = chunks[0]
    if not _topic_present(query, c["text"], idf):
        return None
    if any(not rx.search(c["text"]) for rx, _ in identifiers(query or "")):
        return None
    # The quoted sentence itself must touch the question twice over. One term
    # is not enough: "what is the share price of Infosys today" found a
    # placements page listing Infosys among recruiters and quoted it.
    from rag.scope import content_terms as _ct
    qts = set(_ct(tok(query or "")))
    need = 2 if len(qts) > 1 else 1
    best = [(sc, pos, u) for sc, pos, u in scored_units(query, c["text"], idf)
            if sc >= min_score and _prose(u) and len(qts & set(tok(u))) >= need]
    if not best:
        return None
    picked = sorted(sorted(best, key=lambda x: -x[0])[:max_units], key=lambda x: x[1])
    return " ".join(u.strip() for _, _, u in picked)


def used_from(final, n):
    """Passage indices cited by [n] markers in the answer body."""
    head = final.split(chr(10) + chr(10) + "Sources:")[0]
    return sorted({int(m) - 1 for m in re.findall(r"\[(\d+)\]", head) if 0 < int(m) <= n})
