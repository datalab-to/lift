from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class GenerationResult:
    raw: str
    token_count: int
    error: bool = False


@dataclass
class BatchInputItem:
    images: List[Image.Image]
    schema: dict
    prompt: str | None = None
    prompt_type: str | None = None


@dataclass
class BatchOutputItem:
    extraction: dict
    token_count: int
    error: bool
