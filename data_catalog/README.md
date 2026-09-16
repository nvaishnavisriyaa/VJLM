# Data Catalog

Every source category is classified as **fine-tune / RAG / both / neither**, with a reason, before mass crawling begins. That is the Phase 2 gate.

## Classification rule

Fine-tuning is for stable domain knowledge, VNR terminology and desired answer behaviour. RAG is for authoritative source material, detailed facts, and anything that can change. Stable institutional facts may appear in both. **Frequently changing facts must not be embedded into weights as the primary source of truth.**

| Data type | Fine-tune | RAG | Reason |
|---|---|---|---|
| Stable college identity / history | Yes, carefully | Yes | Domain-native model, authoritative source retained |
| VNR terminology and academic language | Yes | Supporting | Improves domain understanding and answer style |
| Role-specific response behaviour | Yes | No | Behaviour belongs in training, not retrieval |
| Academic regulations / policies | Representative QA only | Yes, full source | Authoritative text gets revised |
| Syllabi / curriculum | Small examples only | Yes | Large, structured, revised |
| Academic calendars / exam schedules | No | Yes | Time-sensitive |
| Faculty names / designations | Generally no | Yes | Changes |
| Placement stats / events / news | No | Yes | Changes frequently |
| Public department information | Stable concepts only | Yes | Mixed stable and changing |
| Login-only personal data | No (Phase 1) | No (Phase 1) | Out of scope |

## Mandatory catalog fields

`catalog.csv` carries one row per source with:

```
source_id, canonical_url, category, department, document_type, audience,
visibility, update_frequency, parser_type, authority_level,
fine_tune_usage, rag_usage, last_seen, last_changed, content_hash,
document_version, active
```
