import json
from typing import List

from nabu.model.schema import BatchInputItem, GenerationResult
from nabu.model.util import scale_to_fit
from nabu.prompts import PROMPT_MAPPING
from nabu.settings import settings


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    **kwargs,
) -> List[GenerationResult]:
    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    conversations = [[process_batch_element(item)] for item in batch]

    inputs = model.processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    # Include both <|endoftext|> and <|im_end|> as stop tokens.
    # generation_config only has <|endoftext|>, but the model emits <|im_end|> at turn boundaries.
    eos_token_id = model.generation_config.eos_token_id
    im_end_id = model.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if im_end_id is not None and im_end_id not in eos_token_id:
        eos_token_id.append(im_end_id)

    generated_ids = model.generate(
        **inputs, max_new_tokens=max_output_tokens, eos_token_id=eos_token_id
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = [
        GenerationResult(raw=out, token_count=len(ids), error=False)
        for out, ids in zip(output_text, generated_ids_trimmed)
    ]
    return results


def process_batch_element(item: BatchInputItem):
    prompt = item.prompt
    prompt_type = item.prompt_type

    if not prompt:
        schema = item.schema
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception:
                pass
        schema_text = json.dumps(schema, indent=2) if isinstance(schema, dict) else str(schema)
        prompt = PROMPT_MAPPING[prompt_type]
        prompt = prompt.replace("{schema}", schema_text)

    content = []
    images = [scale_to_fit(image) for image in item.images]
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def load_model():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        raise ImportError(
            "HuggingFace backend requires additional dependencies. "
            "Install with: pip install nabu-extract[hf]"
        )

    device_map = "auto"
    if settings.TORCH_DEVICE:
        device_map = {"": settings.TORCH_DEVICE}

    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
    }
    if settings.TORCH_ATTN:
        kwargs["attn_implementation"] = settings.TORCH_ATTN

    model = AutoModelForImageTextToText.from_pretrained(
        settings.MODEL_CHECKPOINT, **kwargs
    )
    model = model.eval()
    processor = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    processor.tokenizer.padding_side = "left"
    model.processor = processor
    return model
