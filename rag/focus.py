"""Query-focused passages and completeness supplement (APP-002).

Why this exists
---------------
The fine-tuned model was trained on ONE-passage contexts whose answer is the
passage's first good sentence plus a "Source:" line (training/build_dataset.py:
pick_answer / ctx_block). At inference it is handed five 900-character passages
and does what it was trained to do: it copies the first sentence of the first
passage. Two visible failures followed:

  * the rest of the rule is dropped - for attendance it answered 75% and left
    out the 65-75% condonation band and the "below 65% never condoned" clause;
  * when the first passage is a generic page, a better passage further down
    (the fee guidelines PDF that says Rupay cards are refused) is ignored.

Prompt wording cannot fix a behaviour that was trained in, so two code-side
steps compensate, neither of which lets the model see or say anything that is
not in a retrieved passage:

  focus()       reorders each passage so the sentences that answer THIS question
                come first (headings and unrelated clauses last or dropped);
                the model's "copy the first sentence" habit then copies the
                right one.
  supplement()  after generation, appends the other high-scoring sentences from
                the retrieved passages that the answer did not already say, each
                tagged with the passage it came from.
"""
import re

from rag.index import tok
from rag.scope import content_terms, variants

# clause / sentence boundary: after . ; : ! ? when a numbered clause, bullet or
# capitalised word follows, or at a blank line
_UNIT = re.compile(
    r"(?<=[.;:!?])\s+(?=(?:\d+(?:\.\d+)+[.)]?\s)|[A-Z(•])|\n{2,}|\s+(?=•)")


def units(text):
    raw = [u.strip() for u in _UNIT.split(text or "") if u and u.strip()]
    out, i = [], 0
    while i < len(raw):
        u = raw[i]
        # "14.2 Condonation of Attendance:" is a lead-in, not a sentence: keep it
        # with the sentence it introduces (ALL-CAPS headings stay separate)
        if u.endswith(":") and len(u) < 80 and not _is_heading(u) and i + 1 < len(raw):
            u = u + " " + raw[i + 1]
            i += 1
        out.append(u)
        i += 1
    return out


def _is_heading(u):
    letters = [c for c in u if c.isalpha()]
    return (len(u) < 100 and len(letters) >= 6
            and sum(c.isupper() for c in letters) / len(letters) > 0.85)


def _terms(query, idf):
    ts = []
    for t in content_terms(tok(query)):
        if t not in [x for x, _, _ in ts]:
            ts.append((t, idf.get(t, max(idf.values()) if idf else 1.0), variants(t)))
    return ts


def _score(unit_tokens, terms):
    den = sum(w for _, w, _ in terms) or 1.0
    num = sum(w for t, w, var in terms if t in unit_tokens or any(v in unit_tokens for v in var))
    return num / den


def scored_units(query, text, idf):
    """[(score, position, unit)] for the non-heading units of a passage."""
    terms = _terms(query, idf)
    out = []
    for pos, u in enumerate(units(text)):
        if _is_heading(u):
            continue
        out.append((_score(set(tok(u)), terms), pos, u))
    return out


def focus(query, text, idf, max_chars=650, min_rel=0.25):
    """The passage re-ordered query-first and trimmed to max_chars.

    Units scoring below min_rel of the best unit are dropped (unrelated
    clauses sharing the page). If nothing scores, the original text is used.
    """
    su = scored_units(query, text, idf)
    if not su or max(s for s, _, _ in su) == 0:
        return text[:max_chars]
    best = max(s for s, _, _ in su)
    # Units below half the best score go LAST, not away. Dropping them was
    # costing 14 points: the answer is usually the sentence the question's words
    # are absent from ("in which area of Hyderabad" is answered by a place
    # name), so it scores low and was filtered out of a passage retrieval had
    # correctly selected - answer in the top-5 chunks 83%, answer in the prompt
    # 69%. The filter also made max_chars inert, since it ran before the budget:
    # every surviving unit fit, so the prompt sat at 491 tokens of a 4096
    # context. Ordering query-first is kept (it is what the model answers from);
    # only the hard drop is gone, and leftover budget now carries the rest.
    thr = max(min_rel, 0.5 * best)
    su.sort(key=lambda x: (x[0] < thr, -x[0], x[1]))
    out, n = [], 0
    for s, pos, u in su:
        if n + len(u) > max_chars and out:
            break
        out.append(u)
        n += len(u) + 1
    return "\n".join(out)


# A question asking what is excluded, refused or forbidden.
_NEGATIVE = re.compile(
    r"\b(not allowed|not permitted|not accepted|don'?t work|doesn'?t work|do not work|"
    r"can'?t use|cannot use|disallow\w*|prohibit\w*|forbid\w*|excluded|barred|"
    r"which .{0,30}\bnot\b|what .{0,30}\bnot\b)", re.I)
# A sentence stating one.
_PROHIBIT = re.compile(
    r"\b(not allowed|not permitted|not accepted|not be accepted|shall not|will not be|"
    r"prohibit\w*|forbidden|disallowed|are barred|no longer accepted)\b", re.I)


def _prose(u):
    """Sentence-like text, not a table fragment or a heading."""
    letters = sum(c.isalpha() for c in u)
    return len(u.split()) >= 5 and letters / max(len(u), 1) > 0.6 and u.count("|") < 2


def _flat(u, limit=360):
    u = re.sub(r"\s+", " ", u).strip()
    return u if len(u) <= limit else u[:limit].rsplit(" ", 1)[0] + " ..."


def _prefix(u):
    m = re.match(r"(\d+)\.\d+", u)
    return m.group(1) + "." if m else None


def supplement(query, chunks, answer_body, idf, used=(), max_units=5, min_score=0.6):
    """-> [(chunk_index, unit)] extra sentences, each from a retrieved passage.

    1. Section completion: the clauses that FOLLOW the sentence the answer
       restated, in the same passage and the same numbered clause family
       (14.1 -> 14.2, 14.3 ...). A rule's conditions sit next to its headline
       and rarely share the question's words (condonation, detention).
    2. Question-matching sentences from other passages that the answer did not
       say (the fee guidelines' "Rupay cards are not allowed").
    """
    said = set(tok(answer_body or ""))

    def covered(u):
        ut = [t for t in tok(u) if len(t) > 2 or t.isdigit()]
        return (sum(1 for t in ut if t in said) / len(ut)) if ut else 1.0

    out, seen = [], set()

    def add(ci, u):
        key = " ".join(sorted(set(tok(u))))[:120]
        if key in seen or len(out) >= max_units:
            return
        seen.add(key)
        out.append((ci, _flat(u)))

    # 1. section completion, from the best-supported passage only
    for ci in list(used)[:1]:
        us = units(chunks[ci]["text"])
        best_j, best_cov = None, 0.0
        for j, u in enumerate(us):
            if _is_heading(u) or not _prose(u):
                continue
            c = covered(u)
            if c > best_cov:
                best_j, best_cov = j, c
        pre = _prefix(us[best_j]) if best_j is not None else None
        # numbered clauses only: web pages have no sentence structure to follow
        if best_j is not None and best_cov >= 0.6 and pre:
            taken = 0
            for u in us[best_j + 1:]:
                if _is_heading(u):
                    break
                if not _prose(u) or covered(u) >= 0.7:
                    continue
                if pre and _prefix(u) not in (None, pre):
                    break           # a different numbered clause family starts
                add(ci, u)
                taken += 1
                if taken >= 4:
                    break

    # 1b. the excluding fact, when the question asks for one. "which cards
    # don't work" scores low against "Rupay Credit & Rupay Debit Cards are not
    # allowed": the prohibition shares almost no vocabulary with the question,
    # so the general scorer below never reaches it and the answer came back
    # listing only what IS accepted. Note the corpus disagrees with itself
    # here - the terms-and-conditions PDF lists RuPay as accepted - so both
    # sentences are surfaced with their own [n] rather than either being
    # asserted; a student can see the conflict and check.
    # The bar is lower than the general pass below (a prohibition shares
    # little wording with the question) but it is still the question's own
    # scorer: matching on any shared word pulled a hostel rule about
    # overnight guests into an answer about paying fees.
    if _NEGATIVE.search(query or ""):
        hits = []
        for ci, c in enumerate(chunks):
            for sc, pos, u in scored_units(query, c["text"], idf):
                if sc >= 0.3 and _PROHIBIT.search(u) and _prose(u) and covered(u) < 0.7:
                    hits.append((sc, ci, pos, u))
        for sc, ci, pos, u in sorted(hits, reverse=True)[:2]:
            add(ci, u)

    # 1c. the best-matching sentence of the passage the answer came from, when
    # the answer did not say it. The model copies a passage's FIRST sentence;
    # for "annual tuition fee for Category-A" that was the table caption, while
    # the sentence holding the figure scored 1.0 against the question. Only a
    # near-complete match qualifies, and only one sentence is added.
    for ci in list(used)[:1]:
        best = None
        for sc, pos, u in scored_units(query, chunks[ci]["text"], idf):
            if sc >= 0.9 and _prose(u) and covered(u) < 0.7 and (best is None or sc > best[0]):
                best = (sc, u)
        if best:
            add(ci, best[1])

    # 2. question-matching sentences elsewhere
    cands = []
    for ci, c in enumerate(chunks):
        if ci in used[:1]:
            continue
        for sc, pos, u in scored_units(query, c["text"], idf):
            if sc >= min_score and _prose(u) and covered(u) < 0.7:
                cands.append((sc, -ci, -pos, ci, u))
    cands.sort(reverse=True)
    for sc, _, _, ci, u in cands[:2]:
        add(ci, u)

    out.sort(key=lambda x: (x[0], chunks[x[0]]["text"].find(x[1][:25])))
    return out
