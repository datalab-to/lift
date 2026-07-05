import logging
from typing import List

import filetype
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from lift.settings import settings

logger = logging.getLogger(__name__)


def flatten(page, flag=pdfium_c.FLAT_NORMALDISPLAY) -> bool:
    """Flatten annotations / form fields into the page. Returns True on success."""
    rc = pdfium_c.FPDFPage_Flatten(page, flag)
    return rc != pdfium_c.FLATTEN_FAIL


def load_image(
    filepath: str, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    image = Image.open(filepath).convert("RGB")
    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(
    filepath: str,
    page_range: List[int] | None,
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> List[Image.Image]:
    doc = pdfium.PdfDocument(filepath)
    doc.init_forms()

    images = []
    try:
        for page_index in range(len(doc)):
            if page_range and page_index not in page_range:
                continue

            page_obj = doc[page_index]
            min_page_dim = min(page_obj.get_width(), page_obj.get_height())
            scale_dpi = max((min_pdf_image_dim / min_page_dim) * 72, image_dpi)

            if not flatten(page_obj):
                logger.warning(
                    "Failed to flatten annotations / form fields on page %d.",
                    page_index,
                )

            pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
            images.append(pil_image)
    finally:
        doc.close()
    return images


def parse_range_str(range_str: str) -> List[int]:
    """Parse a page range like '0-5,7,9-12' into a sorted, de-duplicated list of
    page indices. Raises ValueError with a clear message on malformed input."""
    pages = set()
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_str, end_str = part.split("-")
                start, end = int(start_str), int(end_str)
                if start > end:
                    raise ValueError
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
        except ValueError:
            raise ValueError(
                f"Invalid page range segment {part!r} in {range_str!r}. "
                "Use formats like '0-5', '7', or '0-5,7,9-12'."
            ) from None
    return sorted(pages)


def load_file(filepath: str, config: dict) -> List[Image.Image]:
    page_range = config.get("page_range")
    if page_range:
        page_range = parse_range_str(page_range)

    input_type = filetype.guess(filepath)
    if input_type and input_type.extension == "pdf":
        images = load_pdf_images(filepath, page_range)
    else:
        images = [load_image(filepath)]
    return images
