import hashlib
import json
from typing import List

import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image

from nabu.model import InferenceManager
from nabu.input import load_pdf_images
from nabu.model.schema import BatchInputItem
from nabu.schema_builder import (
    LEAF_TYPES,
    delete_schema,
    list_schemas,
    load_schema,
    rows_to_schema,
    save_schema,
    schema_to_rows,
    validate_schema,
)

DEFAULT_ROWS = [
    {"field": "invoice_number", "type": "string", "description": "Invoice identifier", "required": True},
    {"field": "total", "type": "number", "description": "Total amount due", "required": True},
    {"field": "items[].description", "type": "string", "description": "Line item description", "required": False},
    {"field": "items[].amount", "type": "number", "description": "Line item amount", "required": False},
]

ROW_COLUMNS = ["field", "type", "description", "required"]


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


def extract(
    images: List[Image.Image],
    schema: dict,
    model=None,
):
    batch = BatchInputItem(
        images=images,
        prompt_type="direct",
        schema=schema,
    )
    result = model.generate([batch])[0]
    return result


def init_state():
    if "schema_rows" not in st.session_state:
        st.session_state.schema_rows = [dict(r) for r in DEFAULT_ROWS]
        st.session_state.schema_dict = rows_to_schema(DEFAULT_ROWS)
        st.session_state.raw_only = False
        st.session_state.schema_name = ""
        # Bumped whenever the schema is replaced wholesale (load/apply), to reset widgets
        st.session_state.editor_version = 0


def set_schema(schema: dict, name: str = None):
    """Replace the current schema, falling back to raw-only mode if the builder can't represent it."""
    st.session_state.schema_dict = schema
    try:
        st.session_state.schema_rows = schema_to_rows(schema)
        st.session_state.raw_only = False
    except ValueError:
        st.session_state.raw_only = True
    if name is not None:
        st.session_state.schema_name = name
    st.session_state.editor_version += 1


def render_library():
    saved = list_schemas()
    lib_cols = st.columns([0.4, 0.15, 0.3, 0.15])
    with lib_cols[0]:
        selected = st.selectbox(
            "Saved schemas",
            ["—"] + saved,
            label_visibility="collapsed",
            help="Schemas saved in the schemas/ directory.",
        )
    with lib_cols[1]:
        if st.button("Load", disabled=selected == "—", width="stretch"):
            set_schema(load_schema(selected), name=selected)
            st.rerun()
    with lib_cols[2]:
        name = st.text_input(
            "Schema name",
            value=st.session_state.schema_name,
            placeholder="Schema name",
            label_visibility="collapsed",
        )
    with lib_cols[3]:
        if st.button("Save", disabled=not name.strip(), width="stretch"):
            save_schema(name, st.session_state.schema_dict)
            st.session_state.schema_name = name
            st.toast(f"Saved schema '{name}'")
            st.rerun()
    if selected != "—":
        if st.button(f"Delete '{selected}'"):
            delete_schema(selected)
            st.rerun()


def _normalize_editor_rows(df: pd.DataFrame) -> List[dict]:
    """Convert data_editor output to row dicts, skipping rows without a field name."""
    rows = []
    for record in df.to_dict("records"):
        field = record.get("field")
        if field is None or pd.isna(field) or not str(field).strip():
            continue
        row_type = record.get("type")
        description = record.get("description")
        required = record.get("required")
        rows.append({
            "field": str(field).strip(),
            "type": "string" if row_type is None or pd.isna(row_type) else str(row_type),
            "description": "" if description is None or pd.isna(description) else str(description),
            "required": False if required is None or pd.isna(required) else bool(required),
        })
    return rows


def render_builder_tab():
    if st.session_state.raw_only:
        st.info(
            "This schema uses features the table builder can't represent "
            "(e.g. enum, anyOf). Edit it in the Raw JSON tab."
        )
        return

    # The editor input must stay byte-identical across reruns: feeding edits back
    # into the input resets the editor's internal state and drops in-flight edits.
    # schema_rows is only replaced on explicit actions (Load/Apply/Clear), which
    # also bump editor_version to remount the widget.
    edited = st.data_editor(
        pd.DataFrame(st.session_state.schema_rows, columns=ROW_COLUMNS),
        num_rows="dynamic",
        width="stretch",
        key=f"schema_editor_{st.session_state.editor_version}",
        column_config={
            "field": st.column_config.TextColumn(
                "Field",
                help="Dot paths nest objects: customer.name. '[]' makes an array of objects: items[].sku",
                required=True,
            ),
            "type": st.column_config.SelectboxColumn(
                "Type", options=LEAF_TYPES, default="string", required=True
            ),
            "description": st.column_config.TextColumn("Description"),
            "required": st.column_config.CheckboxColumn("Required", default=False),
        },
    )

    if st.button("Clear all fields"):
        st.session_state.schema_rows = []
        st.session_state.schema_dict = {"type": "object", "properties": {}}
        st.session_state.editor_version += 1
        st.rerun()

    rows = _normalize_editor_rows(edited)
    if not rows:
        st.warning("Add at least one field.")
        return

    try:
        schema = rows_to_schema(rows)
    except ValueError as e:
        st.error(str(e))
        return

    error = validate_schema(schema)
    if error:
        st.error(f"Schema failed validation: {error}")
        return

    st.session_state.schema_dict = schema
    st.caption(f"✅ Valid schema ({len(rows)} fields)")
    with st.expander("Generated JSON schema"):
        st.code(json.dumps(schema, indent=2), language="json")


def render_raw_tab():
    # Key by schema content: re-seeds the text area when the schema changes in the
    # builder/library, but stays stable (preserving drafts) while typing here.
    schema_json = json.dumps(st.session_state.schema_dict, indent=2)
    schema_hash = hashlib.md5(schema_json.encode()).hexdigest()[:8]
    raw = st.text_area(
        "JSON schema",
        value=schema_json,
        height=400,
        key=f"raw_schema_{schema_hash}",
        label_visibility="collapsed",
    )
    if st.button("Apply"):
        try:
            schema = json.loads(raw)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            return
        error = validate_schema(schema)
        if error:
            st.error(f"Schema failed validation: {error}")
            return
        set_schema(schema)
        st.rerun()


st.set_page_config(layout="wide", page_title="Nabu Extraction Demo")
init_state()

st.markdown("""
# Nabu Extraction Demo

This app will let you try nabu, a model for structured extraction.
""")

col1, col2 = st.columns([0.5, 0.5])

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

with col1:
    st.markdown("### Schema")
    render_library()
    builder_tab, raw_tab = st.tabs(["Builder", "Raw JSON"])
    with builder_tab:
        render_builder_tab()
    with raw_tab:
        render_raw_tab()

if in_file is None:
    st.stop()

filetype = in_file.type
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

run_extraction = st.sidebar.button("Run Extraction")

if pil_images is None:
    st.stop()

if run_extraction:
    if model_mode == "None":
        st.error("Please select a model mode (hf or vllm) to run extraction.")
        st.stop()

    schema = st.session_state.schema_dict
    error = validate_schema(schema)
    if error:
        st.error(f"Schema failed validation: {error}")
        st.stop()

    with st.spinner("Running extraction..."):
        result = extract(
            pil_images,
            schema,
            model,
        )

    with col1:
        st.markdown("### Result")
        if result.error or result.extraction is None:
            st.error("Extraction failed. Check the server logs for details.")
            if result.raw:
                st.code(result.raw)
        else:
            st.json(result.extraction)
        st.caption(f"{result.token_count} output tokens")

with col2:
    for i, image in enumerate(pil_images):
        st.image(image, caption=f"Image {i}", width="stretch")
