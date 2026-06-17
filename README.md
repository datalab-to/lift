# lift

Extract structured JSON from PDFs and images by passing a schema. lift is a 9B vision model — it reads page images directly (no OCR step) and returns a JSON object matching your schema, with schema-constrained decoding guaranteeing valid output.

## Benchmarks

Evaluated on a 225-document extraction benchmark (6–64 pages per document, ~11,000 scored fields) with adversarial cases planted throughout: cross-page values, exhaustive lists, fields that must be left null, near-miss distractors, multi-source aggregation. Scoring is deterministic exact-match against ground truth (numeric tolerance, normalized strings) - no LLM judging.

All models receive the same rendered page images, and extract each document in a single pass.

| Model | Size | Field accuracy | Full-document accuracy | Median latency* | Features |
|---|---|---|---|---|---|
 | Datalab API | — | 94.7% | 37.3% | 33.1s | Citations + Verification |
| Gemini Flash 3.5 | — | 91.3% | 40.0% | 28.1s | |
| **lift** | 9B | **90.2%** | 20.9% | 9.5s | |
| NuExtract3 | 4B | 81.5% | 8.4% | 8.3s | |
| Qwen3.5-9B | 9B | 76.32% | 24.0% | 16.8s | |

\* Per document, 8 concurrent requests. Local models (lift, Qwen3.5-9B, NuExtract3) served with vLLM on a single GPU; Gemini via API. Latency varies with hardware and load - treat as relative, not absolute.

- **Field accuracy** — fraction of individual schema fields extracted correctly.
- **Full-document accuracy** — fraction of documents where *every* field is correct.
- All models served with default/recommended settings from Github or Huggingface.

Hosted models with verification, citations, and confidence scores are available via the [Datalab API](https://www.datalab.to) — try schemas interactively in the [playground](https://www.datalab.to/playground).
