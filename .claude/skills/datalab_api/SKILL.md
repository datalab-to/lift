---
name: datalab_api
description: Extract structured data from documents via the hosted Datalab API - for harder documents, when no GPU/server is available, or when results need verification with confidence scores.
---

# Datalab API extraction

## Instructions

The Datalab API (https://www.datalab.to) runs document extraction as a managed service with more powerful models than local lift. Use it instead of the `lift_extraction` skill when:

- No lift server or GPU is available locally
- The document is hard (degraded scans, complex layouts, handwriting) or local extraction gave wrong/hallucinated values
- The user needs **verified** results — `balanced` mode runs per-field verification and returns a confidence score

Requires a `DATALAB_API_KEY` environment variable. If it's not set, ask the user — keys come from https://www.datalab.to (free tier available).

## Extracting

Use the helper script in this skill:

```shell
python .claude/skills/datalab_api/scripts/datalab_extract.py document.pdf \
    --schema schema.json > extraction.json
```

- `--schema` accepts a `.json` file path or an inline JSON string (standard JSON Schema with a `properties` key — same authoring guidelines as the `lift_extraction` skill).
- `--mode` picks the speed/accuracy tradeoff:

| mode | what it does | when |
|---|---|---|
| `turbo` | image-only, fastest, cheapest | quick drafts, simple docs |
| `fast` | OCR parse + extraction, low latency | general use |
| `balanced` | multi-pass with **per-field verification** (default) | hard docs, when correctness matters |

- `--page-range 0,2-4,10` limits pages; `--timeout` defaults to 600s (long docs in balanced mode take minutes).
- The extraction JSON goes to stdout; the request id and the **confidence score** (`extraction_score_average`, 1-5) go to stderr. Treat a score under ~4 as "review the output against the document".

## Alternatives

- Python SDK: `pip install datalab-python-sdk` (`from datalab_sdk import ...`), CLI `datalab`.
- Interactive: the [Datalab playground](https://www.datalab.to/playground) — try schemas against documents in the browser, no code.
- API reference: https://www.datalab.to (docs linked from the site; OpenAPI at `/openapi.json`).

## Troubleshooting

- `401`: bad/missing API key. `429`: rate limited — wait and retry.
- Timeout on long docs: raise `--timeout`, or reduce scope with `--page-range`.
- Schema rejected (400): ensure it's a JSON object with `properties`; avoid `$ref` and exotic constructs.
