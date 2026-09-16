# VJLM Benchmark

This is the evidence base for every later "which is better?" decision. It is built in **Phase 5 — before** selecting embedding models, retrieval method, chunking, base SLM, fine-tuning method or quantization.

## Composition

- Balanced student and faculty questions.
- Covers academics, syllabus, regulations, examinations, departments/faculty, placements, fees/scholarships, forms/procedures, clubs/events, facilities and research — wherever public data exists.
- Mix of exact-fact, multi-sentence explanatory, cross-section, and **"not found in public sources"** questions. The last category is what measures abstention instead of fabrication.
- Paraphrases, so retrieval is not rewarded merely for matching a page's exact wording.
- A **hidden held-out split** (`heldout.jsonl`) that is never used to tune anything.

## Record shape

```json
{
  "id": "STU-REG-001",
  "role": "student",
  "question": "...",
  "category": "academic_regulations",
  "gold_source_ids": ["..."],
  "gold_passage": "...",
  "answer_key": "...",
  "requires_retrieval": true,
  "allowed_visibility": ["public"]
}
```

## Metric layers

| Layer | Metrics |
|---|---|
| Retrieval | Recall@k, MRR/nDCG, hit rate, latency |
| Generation | Correctness, faithfulness, completeness, citation accuracy |
| Role behaviour | Role appropriateness, access-policy compliance |
| Runtime | Peak RSS, model size, first-token latency, tokens/sec |
| Maintenance | Update success rate, stale-source rate, crawl/parser errors |

## Rules

1. Do not train on evaluation questions.
2. Split train/validation/test **before** any augmentation, to prevent near-duplicate leakage.
3. Keep the same held-out set across all experiments so comparisons stay fair.
