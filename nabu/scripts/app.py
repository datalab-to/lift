from typing import List

import pypdfium2 as pdfium
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

from nabu.model import InferenceManager
from nabu.input import load_pdf_images
from nabu.model.schema import BatchInputItem


@st.cache_resource()
def load_model(method: str):
    return InferenceManager(method=method)


@st.cache_data()
def get_page_images(pdf_file, range_start, range_end):
    return load_pdf_images(pdf_file, list(range(range_start, range_end)))


@st.cache_data()
def page_counter(pdf_file):
    doc = pdfium.PdfDocument(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def extract(
    images: List[Image.Image],
    schema: dict,
    model=None,
) -> (Image.Image, str):
    batch = BatchInputItem(
        images=images,
        prompt_type="direct",
        schema=schema
    )
    result = model.generate([batch])[0]
    return result


st.set_page_config(layout="wide", page_title="Nabu Extraction Demo")
col1, col2 = st.columns([0.5, 0.5])

st.markdown("""
# Nabu Extraction Demo

This app will let you try nabu, a model for structured extraction.
""")

# Get model mode selection
model_mode = st.sidebar.selectbox(
    "Model Mode",
    ["None", "hf", "vllm"],
    index=0,
    help="Select how to run inference: hf loads the model in memory using huggingface transformers, vllm connects to a running vLLM server.",
)

# Only load model if a mode is selected
model = None
if model_mode == "None":
    st.warning("Please select a model mode (Local Model or vLLM Server) to run extraction.")
else:
    model = load_model(model_mode)

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)
schema = st.sidebar.text(
    "Schema"
)

if in_file is None:
    st.stop()

filetype = in_file.type
page_count = None
pil_images = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_start = st.sidebar.number_input(
        f"Starting page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )
    page_end = st.sidebar.number_input(
        f"Ending page number out of {page_count}:", min_value=1, value=page_count, max_value=page_count
    )

    pil_images = get_page_images(in_file, page_start, page_end)
else:
    pil_images = [Image.open(in_file).convert("RGB")]
    page_start = None
    page_end = None

run_extration = st.sidebar.button("Run Extraction")

if pil_images is None:
    st.stop()

if run_extration:
    if model_mode == "None":
        st.error("Please select a model mode (hf or vllm) to run OCR.")
    else:
        result = extract(
            pil_images,
            schema,
            model,
        )

        with col1:
            st.json(result)

with col2:
    for i, image in enumerate(pil_images):
        st.image(image, caption=f"Image {i}", use_container_width=True)
