import json
import os
from typing import List

from PIL import Image

from lift.input import load_file
from lift.model import InferenceManager
from lift.model.schema import BatchInputItem, BatchOutputItem
from lift.schema_builder import list_schemas, load_schema, validate_schema


def resolve_schema(schema: dict | str) -> dict:
    """Resolve a schema passed as a dict, a path to a JSON file, an inline JSON
    string, or the name of a saved schema in the schema library."""
    if isinstance(schema, dict):
        resolved = schema
    elif os.path.isfile(schema):
        with open(schema) as f:
            resolved = json.load(f)
    elif schema.lstrip().startswith("{"):
        try:
            resolved = json.loads(schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"Schema looks like inline JSON but failed to parse: {e}")
    else:
        try:
            resolved = load_schema(schema)
        except (FileNotFoundError, ValueError):
            available = ", ".join(list_schemas()) or "(none)"
            raise ValueError(
                f"Schema '{schema}' is not a file path, inline JSON, or saved schema name. "
                f"Saved schemas: {available}"
            )

    error = validate_schema(resolved)
    if error:
        raise ValueError(f"Schema failed validation: {error}")
    return resolved


def extract_images(
    images: List[Image.Image],
    schema: dict | str,
    model: InferenceManager,
    max_output_tokens: int | None = None,
    **kwargs,
) -> BatchOutputItem:
    """Extract structured data matching a schema from a list of page images."""
    batch = BatchInputItem(
        images=images,
        prompt_type="direct",
        schema=resolve_schema(schema),
    )
    return model.generate([batch], max_output_tokens=max_output_tokens, **kwargs)[0]


def extract(
    filepath: str,
    schema: dict | str,
    model: InferenceManager | None = None,
    page_range: str | List[int] | None = None,
    max_output_tokens: int | None = None,
    **kwargs,
) -> BatchOutputItem:
    """Run end-to-end extraction on a PDF or image file.

    Args:
        filepath: Path to a PDF or image file.
        schema: JSON schema as a dict, a path to a .json file, an inline JSON
            string, or the name of a saved schema in the schema library.
        model: An InferenceManager to reuse across calls. Created with
            method="vllm" if not provided.
        page_range: Pages to extract from PDFs, as a range string ("0-5,7")
            or a list of page indices.
        max_output_tokens: Maximum number of output tokens.
        **kwargs: Passed through to InferenceManager.generate (e.g. vllm_api_base).
    """
    if model is None:
        model = InferenceManager(method="vllm")

    if isinstance(page_range, (list, tuple)):
        page_range = ",".join(str(p) for p in page_range)
    config = {"page_range": page_range} if page_range else {}
    images = load_file(filepath, config)

    return extract_images(
        images, schema, model, max_output_tokens=max_output_tokens, **kwargs
    )
