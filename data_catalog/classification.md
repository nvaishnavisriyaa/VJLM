# Phase 2 — Fine-Tuning vs RAG Classification

Scope: **Column 1 of the data-source figure — "Data Present on the Website (Publicly available)"**. Columns 2 and 3 (not-on-website, and login-only) are out of Phase 1.

## The rule in one line

**Fine-tuning teaches VJLM how to behave and speak. RAG tells it what is true today.**

If a fact can change — a name, a date, a fee, a rule revision — it belongs in RAG, and only in RAG. If it is about language, framing, or how a student should be answered differently from a faculty member, it belongs in fine-tuning.

## Why almost nothing goes into the weights

Three reasons, in order of how much they bite:

1. **You cannot edit a fact out of a fine-tuned model.** If an HoD changes or a regulation is revised, RAG is a re-index. Weights are a retrain. On Colab, that is hours and a queue.
2. **Personal data must not be memorised.** 487 of the 671 accessible PDFs are faculty CVs. Training on those bakes named individuals' details permanently into a model that will be distributed as a GGUF file. There is no deletion path, which makes any later takedown request impossible to honour. The architecture diagram already calls for PII removal in preprocessing — this is where that starts.
3. **A memorised fact and a retrieved fact look identical to the user,** but only the retrieved one carries a citation. Facts in weights cannot be cited, so they cannot be verified or corrected.

## Category-by-category

| # | Website category (Column 1) | Fine-tune | RAG | Reasoning |
|---|---|---|---|---|
| 1 | **Academics** — syllabus, subjects, credits, electives, Minor/Honors, MEME scheme | Terminology only | **Yes, full** | "What is MEME scheme / CBCS / an Honors track" is vocabulary VJLM should own. The actual course lists and credit counts change per regulation. |
| 2 | **Academic Calendar** — semester dates, mids, holidays | **No** | **Yes** | Purely time-sensitive. A memorised 2026-27 date becomes a confident lie in 2027. |
| 3 | **Academic Regulations** — attendance, detention, CGPA, grading | Representative QA only | **Yes, full source** | Teach the *shape* of a regulation answer (cite the rulebook, state the regulation year). The thresholds themselves stay retrievable — R18/R19/A19/R22/R25 coexist and differ. |
| 4 | **Examinations (Public)** — timetables, notifications, question papers, malpractice rules | **No** for timetables; malpractice rules as QA | **Yes** | Timetables are the most perishable data on the site. Malpractice rules are stable enough to shape answer behaviour. |
| 5 | **Placements (Historical)** — stats, branch-wise, CTC, policy | **No** | **Yes** | Numbers. Never memorise a statistic — a hallucinated CTC figure is the most damaging error this system can make. |
| 6 | **Faculty Information** — names, designations, HoD, email, expertise | **Never** | **Yes** | See reason 2 above. People change roles and leave. |
| 7 | **Departments** — intake, labs, specialisations, PEOs/POs/PSOs | Stable concepts only | **Yes** | "What is a PEO vs a PO vs a PSO" is terminology. The per-department statements are documents. |
| 8 | **Fees & Scholarships** — circulars, schemes, eligibility, deadlines | **No** | **Yes** | Money and deadlines. Highest-consequence category to get wrong. |
| 9 | **Forms & Procedures** — condonation, no-dues, verification | Procedure *shape* as QA | **Yes** | Teaching "answer with the steps, then link the form" is behaviour. Which form and where it lives is retrieval. |
| 10 | **Clubs & Professional Chapters** | **No** | **Yes** | Contacts and office-bearers rotate yearly. |
| 11 | **Events (Published)** | **No** | **Yes** | Time-bound by definition. |
| 12 | **Campus Facilities** — hostel, library, transport, sports | Stable descriptions only | **Yes** | "The college has a central library" is stable. Timings, fees and routes are not. |
| 13 | **Research & Opportunities** — centres, IDEA Lab, VJ Hub, ED Cell, NPTEL | What these entities *are* | **Yes** | Institutional vocabulary in weights; current projects and reports in RAG. |
| 14 | **Grievances & Rights** — Anti-Ragging, ICC, RTI, IQAC | **Yes — behaviour** | **Yes — procedure** | The one category where fine-tuning genuinely earns its place. A student asking about ragging or harassment needs a consistent, careful, non-dismissive tone that always surfaces the official channel. That is a behavioural property, and it must not depend on whether retrieval happened to fire. |

## What the fine-tuning dataset actually is

**Not** scraped website text. A small, hand-verified instruction set of roughly **1,500–3,000 examples**, in five parts:

| Part | Share | Purpose |
|---|---|---|
| Grounded-answer behaviour | ~35% | Given retrieved context, answer from it and cite the source. |
| Abstention | ~20% | Context lacks the answer -> say so. Trained against real VNR-shaped questions whose answers are genuinely absent. |
| Role differentiation | ~20% | Same public regulation, student-facing vs faculty-facing framing. |
| VNR terminology & identity | ~15% | MEME, CBCS, R25, PEO/PO/PSO, IQAC, IDEA Lab, Vignana Jyothi, autonomous status. |
| Sensitive-topic handling | ~10% | Grievance, anti-ragging and ICC questions answered carefully, always routing to the official channel. |

**The abstention and grounding sets matter more than the knowledge set.** A model that says "I could not find this in the available VNRVJIET sources" is more useful to a student than one that invents a plausible fee amount.

## The constraint that makes this expensive

Tier 1 of `INGEST-001` forbids hosted LLM APIs — including at build time. So this dataset **cannot** be mass-generated with GPT or Gemini. The permitted path:

1. **Template-based generation** from structured facts already extracted by the ingestion pipeline. Deterministic, reproducible, and the bulk of the volume.
2. **Local-model-assisted drafting** for paraphrase variety, every output verified before it enters the set.
3. **Human-authored gold examples** for role differentiation and sensitive-topic handling, where tone is the whole point and templates cannot capture it.

This is the largest manual-labour item in the project and should be scheduled as such, not discovered late. Provenance is kept for every example so a bad training row can be traced back to its source.

## Rules carried into Phase 8

- Train/validation/test split happens **before** any augmentation, or paraphrases leak across the boundary.
- The held-out benchmark is never trained on.
- Faculty names, emails and personal details are stripped in preprocessing and never reach the training set.
