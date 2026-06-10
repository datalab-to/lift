---
name: lift_extraction
description: Extract structured data from PDFs and images using the local lift model. Use when the user wants data pulled out of documents into JSON matching a schema.
---

# lift extraction

## Instructions

lift is a vision model that extracts structured JSON from documents. You give it a PDF or image plus a JSON schema; it returns a JSON object matching that schema. Use this skill whenever the user wants structured data out of a document (invoices, papers, forms, statements, contracts, ...).

lift needs an inference backend. Check availability first, in this order:

```shell
curl -s --max-time 3 "${VLLM_API_BASE:-http://localhost:8000/v1}/models"
```

- Response lists a model → server is up, proceed with `--method vllm` (the default).
- No server but the machine has a CUDA GPU → `--method hf` loads weights in-process (requires `pip install lift-pdf[hf]`; downloads ~19GB on first use).
- Neither → don't force it. Tell the user they can start a server with `lift_vllm` on a GPU machine, set `VLLM_API_BASE` to a remote server, or use the `datalab_api` skill (hosted, no GPU needed).

## Writing schemas

Schemas are standard JSON Schema, kept simple:

```json
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": "string", "description": "Invoice identifier"},
    "total": {"type": "number", "description": "Total amount due"},
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "amount": {"type": "number"}
        }
      }
    }
  },
  "required": ["invoice_number", "total"]
}
```

Guidelines:
- Types: `string`, `number`, `integer`, `boolean`, arrays of those, arrays of objects, nested objects.
- Avoid `enum`, `anyOf`/`oneOf`, `$ref`, `additionalProperties` — the schema-constrained decoding skips schemas it can't compile, weakening output guarantees.
- Write `description` for any field whose name isn't self-explanatory — descriptions meaningfully improve accuracy.
- Mark a field `required` only when it must appear in the output; fields genuinely absent from the document should come back `null`.
- Ask the user what fields they want if it isn't obvious from their request. Look at the document first (or its first page) when designing a schema for them.

Reusable schemas live in the repo's `schemas/` directory and can be referenced by name.

## Extracting

CLI (preferred — handles PDFs, images, or whole directories):

```shell
lift_extract document.pdf output/ --schema schema.json
lift_extract scans/ output/ --schema '{"type": "object", "properties": {...}}'
lift_extract doc.pdf output/ --schema invoice --page-range 0-5,8
```

`--schema` accepts a file path, an inline JSON string, or a saved schema name from `schemas/`. Results land in `output/{stem}.json` (the extraction) and `output/{stem}_metadata.json` (pages, tokens, errors).

Python, when extraction is part of a larger script:

```python
from lift import extract

result = extract("document.pdf", schema)  # schema: dict, path, inline JSON, or library name
if result.extraction is not None:
    data = result.extraction  # dict matching the schema
```

Pass `model=InferenceManager(method="hf")` to reuse a loaded model across many calls, and `page_range="0-5"` to limit PDF pages. Set `VLLM_API_BASE` (env or `local.env`) to target a remote server.

## Troubleshooting

- Extraction is `None` / error in metadata: inspect `raw` in `{stem}_metadata.json` — usually truncated generation on very long documents. Retry with `--page-range` to limit pages, or split the document.
- All pages of a document are sent as one request; a 60-page PDF is ~50k tokens and needs a server started with a large `--max-model-len` (the `lift_vllm` launcher handles this).
- Values look wrong or hallucinated, or the user needs verified/cited results: use the `datalab_api` skill instead — its `balanced` mode runs per-field verification.
