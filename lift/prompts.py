DIRECT_EXTRACTION_PROMPT = """Extract structured data from this document according to the provided JSON schema. The document is provided as images, in page order.

## JSON Schema
```json
{schema}
```

## Instructions
- Return a single JSON object that matches the schema exactly.
- Use the correct type for each field (string, number, integer, boolean, array, object).
- Transcribe values as they appear in the document; do not infer, calculate, or invent data.
- A value may span multiple pages — combine the relevant parts into one field.
- If a field is not present in the document, set it to null rather than guessing.
- Return only the JSON object, with no extra commentary."""

PROMPT_MAPPING = {
    "direct": DIRECT_EXTRACTION_PROMPT
}
