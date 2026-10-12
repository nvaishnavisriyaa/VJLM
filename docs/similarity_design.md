# VJLM Similarity Measure — Design

Status: **measured 2026-09-26; two of this document's predictions were wrong.**
Selected measure: `coverage`. Decision and full results in `RAG-SIM-001`.

> **Corrections, kept visible rather than edited away.** This document
> originally assigned cosine the largest weight (γ = 0.40) and argued that
> fusing signals would beat any one alone. The experiment refuted both.
> `dense` finished **last** of eleven arms (MRR 0.509 vs coverage 0.836), and
> **every** fusion lost to `coverage` by itself. The sections below are
> annotated where that happened. A design document that quietly rewrites its
> own predictions after the fact is worth nothing as evidence.

---

## 1. What the measure has to do

Ranking is the obvious job, but for VJLM it is not the only one:

| requirement | why it is here |
|---|---|
| **Rank** candidates for a query | obvious |
| **Decide when nothing is good enough** | abstention is a *core behaviour* of this system, not an edge case. A measure that only ranks cannot say "none of these are relevant" — it always has a top-1. |
| **Bridge vocabulary gaps** | "money help" → *scholarship*, "miss classes" → *attendance shortage* |
| **Respect VNR structure** | R25 vs R22 are different rulebooks. A CSE question should not be answered from the Mechanical syllabus. |
| **Be explainable** | every answer cites sources; "why this document" must be answerable to an examiner |

Requirement 2 is the one that constrains the design most, and it is the one
most similarity measures ignore.

---

## 2. Why a graded term ladder

Sir's rule — **1.0 exact, 0.5 synonym, 0 otherwise** — is the right starting
shape, for a reason worth stating: lexical matching is not binary, and
pretending it is throws away information.

* A literal match is strong evidence.
* A synonym match is real evidence, but weaker — WordNet's sense for a word is
  not guaranteed to be the sense the document meant.
* No match is no evidence, but it is *not* proof of irrelevance.

Keeping those as distinct tiers rather than collapsing to a boolean is what
lets the score stay interpretable: you can always say which tier each query
term landed in.

### The extension: a fourth tier

The ladder has a gap. "Fees" and "tuition" are synonyms and score 0.5. But
"stipend" and "scholarship" are *related without being synonyms*, and WordNet
will not link "shortage" to "detention" even though this corpus treats them as
the same topic. Those currently score 0, which is wrong — they are weak
evidence, not no evidence.

So:

| tier | condition | weight |
|---|---|---|
| exact | the query term occurs in the document | **1.00** |
| synonym | a WordNet/domain synonym occurs | **0.50** |
| **near** | a document term is within `τ` cosine of the query term | **0.25** |
| none | none of the above | **0.00** |

`τ` (default 0.75) and the 0.25 weight are both to be fitted, and the tier is
switchable — if measurement says it only adds noise, it is removed, exactly as
the WordNet verb expansion was.

---

## 3. Term importance: IDF weighting

A plain average over query terms treats `students` and `condonation` as equally
informative. `students` appears in most of the corpus and separates nothing;
`condonation` appears in a handful of documents and separates almost perfectly.

So each term contributes in proportion to its IDF:

```
            Σ_t  idf(t) · tier(t, d)
Cov(q, d) = ────────────────────────        ∈ [0, 1]
                 Σ_t  idf(t)
```

The denominator is what a document would score by containing every query term
literally. That keeps the result a fraction, comparable across queries of
different lengths — which matters for §5.

**Unseen query terms** take the maximum observed IDF. A term absent from the
entire corpus is maximally rare, so it belongs at the top of the scale. It can
never be matched, so it only ever enters the denominator, correctly penalising
every document for not containing something distinctive the user asked about.

---

## 4. The three signals, and why none is sufficient alone

| signal | measures | fails when |
|---|---|---|
| **BM25** | term frequency, saturated, length-normalised | one repeated term dominates; a page repeating "attendance" 50× beats one covering attendance *and* examinations *and* condonation |
| **Coverage** (§2–3) | breadth — how many distinct query concepts appear | cannot see meaning; "miss lectures" shares nothing with "attendance shortage" |
| **Cosine** over embeddings | meaning, whole-query | blurs specifics — R25 and R22 embed almost identically, and exact identifiers (course codes, dates) are where it is weakest |

Measured on the VJLM benchmark (2026-09-26, n=90), all arms on the same
13,051-chunk index:

| mode | Recall@5 | MRR | p50 ms |
|---|---|---|---|
| **coverage** / coverage+syn | **0.944** | **0.836** | 35.0 / 14.0 |
| lexical (0.6·BM25 + 0.4·Cov) | 0.933 | 0.822 | 17.4 |
| bm25+syn | 0.944 | 0.815 | 29.6 |
| bm25 | 0.911 | 0.803 | 19.0 |
| hybrid3 (0.35/0.25/0.40) | 0.900 | 0.780 | 43.6 |
| hybrid (0.5 BM25 + 0.5 cos) | 0.789 | 0.722 | 33.5 |
| dense (cosine alone) | 0.667 | 0.509 | 326.0 |

`coverage` and `coverage+syn` are identical to three decimals: the **0.5
synonym tier never fires** on this benchmark, because the literal word is
always present. The tier is implemented correctly and this instrument cannot
test it.

```
Sim(q, d) = α·BM25̂(q,d)  +  β·Cov(q,d)  +  γ·Cos(q,d)  +  δ·Meta(q,d)
```

All four terms are normalised to [0, 1] so the weights are directly
interpretable as "how much do I trust this signal". `BM25̂` is min-max
normalised **within the candidate pool**, which is a deliberate compromise and
a known weakness — see §5.

Starting point proposed here: `α=0.30, β=0.25, γ=0.40, δ=0.05`.

> **REFUTED (2026-09-26).** `γ=0.40` assumed embeddings would be the strongest
> single signal. Measured on the 90-question benchmark, cosine was the
> *weakest*: MRR 0.509, against 0.836 for coverage and 0.803 for BM25, at 326 ms
> versus 35 ms. Arms were dragged down in proportion to the cosine weight they
> carried — hybrid (0.5) → 0.722, hybrid3 (0.40) → 0.780, cosine-free →
> 0.803–0.836.
>
> **Also refuted: that fusing beats the parts.** Every blend lost to `coverage`
> alone, including `lexical` (0.6·BM25 + 0.4·Cov) at 0.822. On this corpus, one
> good signal beat every mixture tried.
>
> **What this does and does not license.** The benchmark's questions are ~12
> words lifted verbatim from the gold passage — a bag of exact keywords, not a
> paraphrase. Coverage is *defined* as exact keyword overlap, so it was measured
> on its strength; cosine was asked for semantic matching on a query needing
> none. The supported claim is narrow: **on keyword-style queries, lexical wins
> and coverage is the best lexical measure.** Whether embeddings help on real
> student phrasing is still unmeasured — `evaluation/paraphrase_probe.py` shows
> every lexical arm returning faculty CVs for "what happens if I miss too many
> lectures", which no keyword weighting fixes.

---

## 5. Absolute calibration — the part that enables abstention

This is the design decision that matters most and is easiest to get wrong.

BM25 scores are unbounded and corpus-relative; min-max normalising them within
a pool makes the *best* candidate score 1.0 **whether or not it is any good**.
A system that abstains based on that will never abstain. Every query has a
winner.

So the abstention gate must be computed from the signals that are absolutely
scaled:

```
Confidence(q, d) = β'·Cov(q, d) + γ'·Cos(q, d)          (β' + γ' = 1)
```

Both are genuinely bounded: coverage is a fraction of achievable IDF mass,
cosine is a similarity on unit vectors. Neither depends on what else happened
to be in the pool.

```
if max_d Confidence(q, d) < θ_abstain:
        → "I could not find this in the available VNRVJIET sources."
```

`θ_abstain` is fitted on the **unanswerable** questions already in the
evaluation set — the "what are my internal marks" class, whose answers are
genuinely not in public data. That gives a labelled negative set, which is
exactly what is needed to place a threshold and is why those questions were
written in the first place.

Two properties to verify when fitting:

1. **Ranking and abstention are separate concerns.** Reordering the pool must
   not change whether the system abstains.
2. **The threshold must be checked against the fine-tuned model's own
   abstention.** The model is already trained to refuse when context is
   insufficient. If the retriever also abstains, the two must not double-count
   into refusing answerable questions. Measure false-abstention rate, which
   `evaluation/eval_generation.py` already reports.

---

## 6. Metadata alignment (`Meta`)

Small weight, high precision. This is VNR-specific structure that no
general-purpose measure knows about:

| condition | contribution |
|---|---|
| query names a regulation (R25/R22/A19) and the chunk carries it | +1.0 |
| query names a department and the chunk matches | +1.0 |
| query implies currency ("this year") and the chunk is the latest year | +0.5 |
| chunk's regulation **contradicts** one named in the query | **−1.0** |

The negative case is the important one. Retrieving the R22 attendance rule for
a student who said "R25" is worse than retrieving nothing — it is a confident,
plausible, wrong answer, and it will be cited as authoritative.

---

## 7. What gets measured before any of this is adopted

| experiment | question it settles |
|---|---|
| coverage vs BM25 vs cosine, alone | which single signal is strongest |
| tier ablation: 1/0.5/0 vs 1/0.5/0.25/0 vs 1/0/0 | does the synonym tier earn its place; does `near` |
| synonym weight sweep 0.3 / 0.5 / 0.7 | is 0.5 right for this corpus |
| weight grid over α, β, γ | the fusion |
| θ_abstain sweep on unanswerable questions | precision/recall of abstention |

**Every one of these needs the Phase 5 human benchmark to be trustworthy.** The
current 90 questions are auto-generated by lifting words verbatim from the gold
passage, which structurally favours coverage and cannot exercise the synonym
tier at all — `coverage` and `coverage+syn` scored *identically* because the
literal words were always present. That benchmark can rank lexical variants
against each other; it cannot validate a semantic measure.

---

## 8. Honest summary

| piece | status |
|---|---|
| Graded tier ladder, IDF-weighted (`rag/coverage.py`) | built, measured, beats BM25 |
| Synonym tier (`rag/synonyms.py`) | built; **untestable** on the current benchmark |
| `near` tier | designed, not built — needs embeddings |
| Cosine arm | **built and measured** — `dense.npy` (13,051 × 384); finished last |
| Fusion weights | guessed, **measured, and beaten by a single signal** |
| Abstention threshold | designed, not implemented |
| Metadata alignment | designed, not implemented |

The embeddings blocker is cleared — `dense.npy` exists and cosine has been
measured. What replaces it is sharper and harder: **the benchmark can only
measure keyword overlap.** Its questions are built from the gold passage's own
words, so it rewards the measure that counts those words and penalises the one
that does not need them. Every number in this document rests on that
instrument.

The Phase 5 human benchmark is therefore no longer a nice-to-have. It is the
only way to learn whether `dense = 0.509` is a real property of this corpus or
an artefact of how 90 questions were generated — and the only way the 0.5
synonym tier can be exercised at all.
