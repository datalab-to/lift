DIRECT_EXTRACTION_PROMPT = """Extract structured data from this document according to the provided JSON schema.  The document is provided as images, in page order.

## JSON Schema
```json
{schema}
```

## Instructions
- Return a JSON object matching the schema
- Use the correct type for each field (string, number, array)"""

PROMPT_MAPPING = {
    "direct": DIRECT_EXTRACTION_PROMPT
}