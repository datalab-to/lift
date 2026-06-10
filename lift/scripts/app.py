import copy
import hashlib
import json
import time
from datetime import datetime
from typing import List

import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image

from lift.extract import extract_images
from lift.model import InferenceManager
from lift.input import load_pdf_images
from lift.schema_builder import (
    LEAF_TYPES,
    delete_schema,
    list_schemas,
    load_schema,
    rows_to_schema,
    save_schema,
    schema_to_rows,
    validate_schema,
)

DATALAB_URL = "https://www.datalab.to"
PLAYGROUND_URL = "https://www.datalab.to/playground"
MAX_HISTORY = 20

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


def schema_fingerprint(schema: dict) -> str:
    """Short content hash, used to tell schema versions apart across test runs."""
    return hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:8]


def init_state():
    if "schema_rows" not in st.session_state:
        st.session_state.schema_rows = [dict(r) for r in DEFAULT_ROWS]
        st.session_state.schema_dict = rows_to_schema(DEFAULT_ROWS)
        st.session_state.raw_only = False
        st.session_state.schema_name = ""
        # Bumped whenever the schema is replaced wholesale (load/apply), to reset widgets
        st.session_state.editor_version = 0
        st.session_state.run_history = []
        st.session_state.import_digest = None


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


# ── Schema library ───────────────────────────────────────────────────────────


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
            existed = name in saved
            save_schema(name, st.session_state.schema_dict)
            st.session_state.schema_name = name
            st.toast(f"{'Updated' if existed else 'Saved'} schema '{name}'")
            st.rerun()

    io_cols = st.columns([0.25, 0.25, 0.5])
    with io_cols[0]:
        st.download_button(
            "Download JSON",
            json.dumps(st.session_state.schema_dict, indent=2),
            file_name=f"{st.session_state.schema_name or 'schema'}.json",
            mime="application/json",
            width="stretch",
            disabled=not st.session_state.schema_dict.get("properties"),
        )
    with io_cols[1]:
        with st.popover("Import", width="stretch"):
            uploaded = st.file_uploader("Schema JSON", type=["json"], key="schema_import")
            if uploaded is not None:
                data = uploaded.getvalue()
                digest = hashlib.md5(data).hexdigest()
                if digest != st.session_state.import_digest:
                    try:
                        schema = json.loads(data)
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                        schema = None
                    if schema is not None:
                        error = validate_schema(schema)
                        if error:
                            st.error(f"Schema failed validation: {error}")
                        else:
                            st.session_state.import_digest = digest
                            set_schema(schema)
                            st.rerun()
    with io_cols[2]:
        if selected != "—" and st.button(f"Delete '{selected}'"):
            delete_schema(selected)
            st.rerun()


# ── Schema editing ───────────────────────────────────────────────────────────


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
    raw = st.text_area(
        "JSON schema",
        value=schema_json,
        height=400,
        key=f"raw_schema_{schema_fingerprint(st.session_state.schema_dict)}",
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


# ── Schema testing ───────────────────────────────────────────────────────────


def _leaf_status(value) -> str:
    if value is None or value == "" or value == []:
        return "empty"
    return "filled"


def _preview(value, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _lookup_path(extraction: dict, field: str) -> tuple[str, str]:
    """Resolve a builder field path (dots nest, [] fans out over arrays) in an
    extraction. Returns (status, preview) with status in filled/empty/missing."""
    node = extraction
    segments = field.split(".")
    for i, seg in enumerate(segments):
        is_array = seg.endswith("[]")
        key = seg[:-2] if is_array else seg
        if not isinstance(node, dict) or key not in node:
            return "missing", ""
        node = node[key]
        if is_array:
            rest = ".".join(segments[i + 1 :])
            if not isinstance(node, list):
                return ("empty", "") if node in (None, []) else ("filled", _preview(node))
            if not node:
                return "empty", "[]"
            statuses, previews = [], []
            for item in node:
                if rest:
                    s, p = _lookup_path(item if isinstance(item, dict) else {}, rest)
                else:
                    s, p = _leaf_status(item), _preview(item)
                statuses.append(s)
                if s == "filled":
                    previews.append(p)
            filled = sum(1 for s in statuses if s == "filled")
            status = "filled" if filled else ("empty" if "empty" in statuses else "missing")
            return status, f"{filled}/{len(node)} items · " + ", ".join(previews[:3])
    return _leaf_status(node), _preview(node)


def field_report(schema: dict, extraction: dict) -> List[dict] | None:
    """Per-leaf fill report for the test run. None when the schema can't be
    represented as builder rows (raw-only) — caller falls back to JSON view."""
    try:
        rows = schema_to_rows(schema)
    except ValueError:
        return None
    report = []
    for row in rows:
        status, preview = _lookup_path(extraction, row["field"])
        if status == "filled":
            icon = "✅"
        elif row["required"]:
            icon = "❌" if status == "missing" else "⚠️"
        else:
            icon = "–"
        report.append({
            "field": row["field"],
            "required": row["required"],
            "status": f"{icon} {status}",
            "value": preview,
        })
    return report


def coverage(report: List[dict]) -> tuple[int, int]:
    filled = sum(1 for r in report if r["status"].endswith("filled") and "✅" in r["status"])
    return filled, len(report)


def record_run(schema: dict, result, doc_name: str, pages: str, latency: float):
    failed = result.error or result.extraction is None
    report = None if failed else field_report(schema, result.extraction)
    run = {
        "run": len(st.session_state.run_history) + 1,
        "time": datetime.now().strftime("%H:%M:%S"),
        "schema": copy.deepcopy(schema),
        "schema_name": st.session_state.schema_name or "(unsaved)",
        "schema_hash": schema_fingerprint(schema),
        "doc": doc_name,
        "pages": pages,
        "latency": latency,
        "tokens": result.token_count,
        "error": bool(failed),
        "extraction": result.extraction,
        "raw": result.raw if failed else None,
        "report": report,
    }
    st.session_state.run_history.append(run)
    del st.session_state.run_history[:-MAX_HISTORY]


def _previous_coverage(run) -> tuple[int, int] | None:
    """Most recent earlier run on the same document with a report."""
    for prev in reversed(st.session_state.run_history):
        if prev["run"] < run["run"] and prev["doc"] == run["doc"] and prev["report"]:
            return coverage(prev["report"])
    return None


def render_results(run):
    st.markdown(f"### Result — run #{run['run']}")
    st.caption(
        f"{run['tokens']} output tokens · {run['latency']:.1f}s · "
        f"schema {run['schema_name']} ({run['schema_hash']}) · {run['doc']} p{run['pages']}"
    )

    if run["error"]:
        st.error("Extraction failed. Check the server logs for details.")
        if run["raw"]:
            with st.expander("Raw model output"):
                st.code(run["raw"])
        st.caption(
            f"Hard document? The [Datalab playground]({PLAYGROUND_URL}) runs more "
            f"powerful models with per-field verification."
        )
        return

    if run["report"]:
        filled, total = coverage(run["report"])
        delta = _previous_coverage(run)
        delta_text = f" (was {delta[0]}/{delta[1]})" if delta and delta != (filled, total) else ""
        st.markdown(f"**Coverage: {filled}/{total} fields filled{delta_text}**")
        st.dataframe(pd.DataFrame(run["report"]), width="stretch", hide_index=True)
        with st.expander("Extraction JSON"):
            st.json(run["extraction"])
    else:
        # Raw-only schema: no per-field report, show the extraction directly.
        st.json(run["extraction"])


def render_history():
    history = st.session_state.run_history
    if len(history) < 2:
        return
    with st.expander(f"Run history ({len(history)})"):
        table = []
        for run in history:
            cov = coverage(run["report"]) if run["report"] else None
            table.append({
                "run": run["run"],
                "time": run["time"],
                "schema": f"{run['schema_name']} ({run['schema_hash']})",
                "doc": run["doc"],
                "pages": run["pages"],
                "coverage": f"{cov[0]}/{cov[1]}" if cov else ("error" if run["error"] else "—"),
                "tokens": run["tokens"],
                "latency": f"{run['latency']:.1f}s",
            })
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

        inspect_cols = st.columns([0.3, 0.35, 0.35])
        with inspect_cols[0]:
            run_no = st.selectbox("Inspect run", [r["run"] for r in history])
        selected_run = next(r for r in history if r["run"] == run_no)
        with inspect_cols[1]:
            if st.button("Restore this run's schema", width="stretch"):
                name = selected_run["schema_name"]
                set_schema(
                    copy.deepcopy(selected_run["schema"]),
                    name=None if name == "(unsaved)" else name,
                )
                st.rerun()
        with inspect_cols[2]:
            if st.button("Clear history", width="stretch"):
                st.session_state.run_history = []
                st.rerun()
        if selected_run["extraction"] is not None:
            st.json(selected_run["extraction"], expanded=False)


# ── Page ─────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="lift Schema Studio")
init_state()

st.markdown("""
# lift Schema Studio

Build, save, and test extraction schemas against your documents.
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
    st.warning(
        "Select a model mode (hf or vllm) in the sidebar to test schemas locally — "
        f"or try them instantly in the [Datalab playground]({PLAYGROUND_URL}) "
        "(no setup, more powerful models)."
    )
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

# Sidebar footer with Datalab links (kept quiet; rendered before st.stop paths)
st.sidebar.divider()
st.sidebar.markdown(
    f"Built on [lift by Datalab]({DATALAB_URL}). Need more accurate models, "
    f"verification, and citations? Try the [hosted playground]({PLAYGROUND_URL})."
)

if in_file is None:
    with col1:
        if st.session_state.run_history:
            render_results(st.session_state.run_history[-1])
            render_history()
    st.stop()

filetype = in_file.type
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_start = st.sidebar.number_input(
        f"First page (0-indexed, document has {page_count}):",
        min_value=0,
        value=0,
        max_value=page_count - 1,
    )
    page_end = st.sidebar.number_input(
        f"Last page (exclusive):",
        min_value=page_start + 1,
        value=page_count,
        max_value=page_count,
    )

    pil_images = get_page_images(in_file, page_start, page_end)
    pages_label = f"{page_start}-{page_end}"
else:
    pil_images = [Image.open(in_file).convert("RGB")]
    pages_label = "image"

run_extraction = st.sidebar.button("Test schema on document")

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
        start = time.monotonic()
        result = extract_images(pil_images, schema, model)
        elapsed = time.monotonic() - start

    record_run(schema, result, in_file.name, pages_label, elapsed)

with col1:
    if st.session_state.run_history:
        render_results(st.session_state.run_history[-1])
        render_history()

with col2:
    for i, image in enumerate(pil_images):
        st.image(image, caption=f"Image {i}", width="stretch")
