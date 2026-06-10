import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import List

from PIL import Image
from json_schema_to_pydantic import create_model
from openai import OpenAI

from lift.model.schema import BatchInputItem, GenerationResult
from lift.model.util import scale_to_fit, detect_repeat_token
from lift.prompts import PROMPT_MAPPING
from lift.settings import settings


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


_NULLABLE_LEAF_TYPES = ("string", "number", "integer", "boolean")


def make_properties_nullable(node):
    """Allow null for every property leaf in a JSON schema (in place).

    Without this, schema-constrained decoding grammar-forces a typed value
    for every field, so the model literally cannot abstain on fields absent
    from the document — it hallucinates a value (or the string "null")
    instead. Allowing null lifts field accuracy and roughly quadruples
    should-be-null accuracy on the extraction benchmark, at no latency cost.
    Object/array structure and array item types are left untouched.
    """
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            for spec in props.values():
                if isinstance(spec, dict):
                    leaf_type = spec.get("type")
                    if isinstance(leaf_type, str) and leaf_type in _NULLABLE_LEAF_TYPES:
                        spec["type"] = [leaf_type, "null"]
        for value in node.values():
            make_properties_nullable(value)
    elif isinstance(node, list):
        for value in node:
            make_properties_nullable(value)


def generate_vllm(
    batch: List[BatchInputItem],
    max_output_tokens: int = None,
    max_retries: int = None,
    max_workers: int | None = None,
    custom_headers: dict | None = None,
    max_failure_retries: int | None = None,
    vllm_api_base: str = settings.VLLM_API_BASE,
    temperature: float = 0.0,
    top_p: float = 0.1,
) -> List[GenerationResult]:
    client = OpenAI(
        api_key=settings.VLLM_API_KEY,
        base_url=vllm_api_base,
        default_headers=custom_headers,
    )
    model_name = settings.VLLM_MODEL_NAME

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    if max_workers is None:
        max_workers = min(64, len(batch))

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if model_name is None:
        models = client.models.list()
        model_name = models.data[0].id

    def _generate(item: BatchInputItem, temperature, top_p) -> GenerationResult:
        schema = item.schema
        kwargs = {}
        try:
            if isinstance(schema, str):
                schema = json.loads(schema)
            schema_model = create_model(schema)
            json_schema = schema_model.model_json_schema()
            make_properties_nullable(json_schema)
            schema_dict = {
                "name": schema_model.__name__,
                "schema": json_schema,
            }

            # Enforce schema guardrails in vllm
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": schema_dict,
            }
        except Exception as e:
            print(f"Schema failed validation with error {e}, skipping guardrails...")

        prompt = item.prompt
        if not prompt:
            schema_text = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)
            prompt = PROMPT_MAPPING[item.prompt_type]
            prompt = prompt.replace("{schema}", schema_text)

        content = []
        images = [scale_to_fit(image) for image in item.images]
        images_b64 = [image_to_base64(image) for image in images]
        for image_b64 in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )

        content.append({"type": "text", "text": prompt})

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            raw = completion.choices[0].message.content
            result = GenerationResult(
                raw=raw,
                token_count=completion.usage.completion_tokens,
                error=False,
            )
        except Exception as e:
            print(f"Error during VLLM generation: {e}")
            return GenerationResult(raw="", token_count=0, error=True)

        return result

    def process_item(item, max_retries, max_failure_retries=None):
        result = _generate(item, temperature=temperature, top_p=top_p)
        retries = 0

        while _should_retry(result, retries, max_retries, max_failure_retries):
            retry_temperature = min(temperature + 0.2 * (retries + 1), 0.8)
            result = _generate(item, temperature=retry_temperature, top_p=0.95)
            retries += 1

        return result

    def _should_retry(result, retries, max_retries, max_failure_retries):
        has_repeat = detect_repeat_token(result.raw) or (
            len(result.raw) > 50 and detect_repeat_token(result.raw, cut_from_end=50)
        )

        if retries < max_retries and has_repeat:
            print(
                f"Detected repeat token, retrying generation (attempt {retries + 1})..."
            )
            return True

        if retries < max_retries and result.error:
            print(
                f"Detected vllm error, retrying generation (attempt {retries + 1})..."
            )
            time.sleep(2 * (retries + 1))  # Sleeping can help under load
            return True

        if (
            result.error
            and max_failure_retries is not None
            and retries < max_failure_retries
        ):
            print(
                f"Detected vllm error, retrying generation (attempt {retries + 1})..."
            )
            time.sleep(2 * (retries + 1))  # Sleeping can help under load
            return True

        return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                process_item, batch, repeat(max_retries), repeat(max_failure_retries)
            )
        )

    return results
