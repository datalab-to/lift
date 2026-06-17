<p align="center">
  <img src="assets/datalab-logo.png" alt="Datalab Logo" width="150"/>
</p>
<h1 align="center">Datalab</h1>
<p align="center">
  <strong>State of the Art models for Document Intelligence</strong>
</p>
<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg" alt="Code License"></a>
  <a href="https://www.datalab.to/pricing"><img src="https://img.shields.io/badge/Model%20License-OpenRAIL--M-blue.svg" alt="Model License"></a>
  <a href="https://discord.gg/KuZwXNGnfH"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
</p>
<p align="center">
  <a href="https://www.datalab.to"><img src="https://img.shields.io/badge/Homepage-datalab.to-blue" alt="Homepage"></a>
  <a href="https://documentation.datalab.to"><img src="https://img.shields.io/badge/Docs-Read%20the%20docs-blue" alt="Docs"></a>
  <a href="https://www.datalab.to/playground"><img src="https://img.shields.io/badge/Playground-Try%20it-orange" alt="Public Playground"></a>
</p>

<hr/>

# lift

lift extracts structured JSON from PDFs and images by passing a schema. It's a 9B vision model — it reads page images directly (no OCR step) and returns a JSON object matching your schema, with schema-constrained decoding guaranteeing valid output.

## Try lift on Datalab

Our managed platform runs improved extraction with higher accuracy than the open weights, plus per-field verification, citations, and confidence scores — zero data retention by default, SOC 2 Type 2, and custom BAAs.

If you have high volume workloads, we offer a batch processing service that has processed 200M+ pages per week — we manage the infrastructure so your workloads finish on time.

Get started with **$5 in free credits** — [sign up](https://www.datalab.to/?utm_source=gh-lift) — takes under 30 seconds — or try lift in our [public playground](https://www.datalab.to/playground?utm_source=gh-lift).

Commercial self-hosting requires a license — see [Commercial usage](#commercial-usage). For on-prem licensing, [contact us](https://www.datalab.to/contact?utm_source=gh-lift-onprem).

## Features

- Extract structured data straight from page images — no separate OCR step
- Pass any JSON schema; **schema-constrained decoding guarantees valid, well-typed output**
- Nullable fields by default, so the model abstains on data that isn't in the document instead of hallucinating
- Handles multi-page documents in a single pass, including values that span pages
- Two inference modes: local (HuggingFace) and remote (vLLM server)
- CLI for single files, inline schemas, or whole directories
- Schema Studio: a Streamlit app to build, save, and test schemas against your documents

## Quickstart

The easiest way to start is with the CLI tools:

```shell
pip install lift-pdf

# With vLLM (recommended, lightweight install)
lift_vllm
lift_extract input.pdf ./output --schema schema.json

# With HuggingFace (requires torch)
pip install lift-pdf[hf]
lift_extract input.pdf ./output --schema schema.json --method hf
```

## Benchmarks

Evaluated on a 225-document extraction benchmark (6–64 pages per document, ~11,000 scored fields) with adversarial cases planted throughout: cross-page values, exhaustive lists, fields that must be left null, near-miss distractors, multi-source aggregation. Scoring is deterministic exact-match against ground truth (numeric tolerance, normalized strings) — no LLM judging.

All models receive the same rendered page images, capped at 861,696 pixels per page, and extract each document in a single pass.

| Model | Size | Field accuracy | Full-document accuracy | Median latency* |
|---|---|---|---|---|
| **lift** | 9B | **90.2%** | 20.9% | 9.5s |
| Qwen3.5-9B | 9B | 89.9% | 23.1% | 9.9s |
| NuExtract3 | 4B | 81.5% | 8.4% | 8.3s |
| Gemini Flash 3.5 (minimal reasoning) | — | 93.4% | 32.0% | 14.6s |
| Gemini Flash 3.5 (default reasoning) | — | 91.3% | 40.0% | 28.1s |

\* Per document, 8 concurrent requests. Local models (lift, Qwen3.5-9B, NuExtract3) served with vLLM on a single GPU; Gemini via API. Latency varies with hardware and load — treat as relative, not absolute.

- **Field accuracy** — fraction of individual schema fields extracted correctly.
- **Full-document accuracy** — fraction of documents where *every* field is correct.
- lift and Qwen3.5-9B use schema-constrained decoding with nullable fields (lift's default inference path); NuExtract3 uses its native template format; Gemini receives the schema in-prompt.

Hosted models with verification, citations, and confidence scores are available via the [Datalab API](https://www.datalab.to) — try schemas interactively in the [playground](https://www.datalab.to/playground).

## Installation

### Package

```bash
# Base install (for vLLM backend)
pip install lift-pdf

# With HuggingFace backend (includes torch, transformers)
pip install lift-pdf[hf]

# With the Schema Studio app
pip install lift-pdf[app]

# With all extras
pip install lift-pdf[all]
```

If you're using the HuggingFace method, we also recommend installing [flash attention](https://github.com/Dao-AILab/flash-attention) for better performance.

### From Source

```bash
git clone https://github.com/datalab-to/lift.git
cd lift
uv sync
source .venv/bin/activate
```

## Usage

### Schemas

A schema is standard JSON Schema. Keep it simple — `string`, `number`, `integer`, `boolean`, arrays of those, arrays of objects, and nested objects are all supported. Avoid `enum`, `anyOf`/`oneOf`, `$ref`, and `additionalProperties`; the schema-constrained decoder skips schemas it can't compile, which weakens the output guarantee.

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

Write a `description` for any field whose name isn't self-explanatory — descriptions meaningfully improve accuracy. Mark a field `required` only when it must appear; fields genuinely absent from a document come back `null`.

### CLI

Process single files or entire directories:

```bash
# Single file, with the vLLM server (see below for how to launch it)
lift_extract input.pdf ./output --schema schema.json

# Inline JSON schema
lift_extract scans/ ./output --schema '{"type": "object", "properties": {...}}'

# A schema saved by name in the schemas/ directory, limited to some pages
lift_extract input.pdf ./output --schema invoice --page-range 0-5,8

# Process a whole directory with the local HuggingFace model
lift_extract ./documents ./output --schema schema.json --method hf
```

**CLI Options:**
- `--schema TEXT` (required): a path to a JSON schema file, an inline JSON string, or the name of a saved schema in the schema library.
- `--method [hf|vllm]`: inference method (default: `vllm`).
- `--page-range TEXT`: page range for PDFs, e.g. `"0-5,7,9-12"` (PDFs only).
- `--max-output-tokens INTEGER`: maximum number of output tokens.

**Output Structure:**

For each processed file, `lift_extract` writes to the output directory:
- `<filename>.json` — the extraction matching your schema
- `<filename>_metadata.json` — page count, token count, and error info (with the raw model output when extraction fails, for debugging)

### Python

```python
from lift import extract

# schema: a dict, a path to a .json file, an inline JSON string, or a library name
result = extract("document.pdf", "schema.json")
if result.extraction is not None:
    data = result.extraction  # dict matching the schema
```

Pass `model=InferenceManager(method="hf")` to load weights in-process and reuse them across calls, and `page_range="0-5"` to limit PDF pages. Set `VLLM_API_BASE` to target a remote server.

### Schema Studio

Launch the interactive app to build, save, and test extraction schemas against your documents (requires `pip install lift-pdf[app]`):

```bash
lift_app
```

### vLLM Server

For production deployments or batch processing, launch the vLLM server:

```bash
lift_vllm                # defaults to H100 settings
lift_vllm --gpu a100-80  # tune batch settings for your GPU
```

This launches a Docker container with optimized inference settings, automatically scaling batch size to your GPU's VRAM. Supported GPUs: `h100`, `a100-80`, `a100`/`a100-40`, `l40s`, `a10`, `l4`, `4090`, `3090`, `t4`.

You can also start your own vLLM server with the `datalab-to/lift-oss-0.1.7` model.

### Configuration

Settings can be configured via environment variables or a `local.env` file:

```bash
# Model settings
MODEL_CHECKPOINT=datalab-to/lift-oss-0.1.7
MAX_OUTPUT_TOKENS=12384
TORCH_DEVICE=cuda:0     # pin the HF backend to a device

# vLLM settings
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL_NAME=lift
VLLM_GPUS=0
```

# Commercial usage

This code is Apache 2.0, and our model weights use a modified OpenRAIL-M license (free for research, personal use, and startups under $2M funding/revenue, cannot be used competitively with our API). To remove the OpenRAIL license requirements, or for broader commercial licensing, visit our pricing page [here](https://www.datalab.to/pricing?utm_source=gh-lift).

# Credits

Thank you to the following open source projects:

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen 3.5](https://github.com/QwenLM/Qwen3)
