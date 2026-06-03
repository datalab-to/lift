import json
import os
import re
from typing import List

from json_schema_to_pydantic import create_model

from lift.settings import settings

# Leaf types selectable in the builder. Objects and arrays of objects are
# implied by dot paths (customer.name) and [] segments (items[].sku).
LEAF_TYPES = ["string", "number", "integer", "boolean", "string[]", "number[]", "integer[]", "boolean[]"]

_SEGMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\[\])?$")


def _leaf_schema(row_type: str, description: str) -> dict:
    if row_type.endswith("[]"):
        leaf = {"type": "array", "items": {"type": row_type[:-2]}}
    else:
        leaf = {"type": row_type}
    if description:
        leaf["description"] = description
    return leaf


def rows_to_schema(rows: List[dict]) -> dict:
    """Convert builder rows into a JSON schema.

    Each row: {"field": "items[].sku", "type": "string", "description": "...", "required": bool}
    """
    schema = {"type": "object", "properties": {}, "required": []}
    seen = set()

    for row in rows:
        field = (row.get("field") or "").strip()
        row_type = row.get("type") or "string"
        description = (row.get("description") or "").strip()
        required = bool(row.get("required"))

        if not field:
            raise ValueError("A row has an empty field name.")
        if field in seen:
            raise ValueError(f"Duplicate field: '{field}'")
        seen.add(field)
        if row_type not in LEAF_TYPES:
            raise ValueError(f"Field '{field}' has unknown type '{row_type}'.")

        segments = field.split(".")
        if not all(_SEGMENT_RE.match(seg) for seg in segments):
            raise ValueError(
                f"Invalid field path '{field}'. Use names like 'total', 'customer.name', or 'items[].sku'."
            )
        if segments[-1].endswith("[]"):
            raise ValueError(
                f"Field '{field}' ends with '[]'. Array-of-object segments need a child field "
                f"(e.g. '{field}.value'); for a plain list use a '{row_type}[]'-style type instead."
            )

        node = schema
        for seg in segments[:-1]:
            name = seg[:-2] if seg.endswith("[]") else seg
            child = node["properties"].get(name)
            if child is None:
                child = {"type": "object", "properties": {}, "required": []}
                if seg.endswith("[]"):
                    node["properties"][name] = {"type": "array", "items": child}
                else:
                    node["properties"][name] = child
            else:
                # Walk into an existing object or array-of-object node
                is_array = child.get("type") == "array"
                if is_array != seg.endswith("[]") or (is_array and child.get("items", {}).get("type") != "object"):
                    raise ValueError(f"Field '{field}' conflicts with an earlier definition of '{name}'.")
                if is_array:
                    child = child["items"]
                if child.get("type") != "object":
                    raise ValueError(f"Field '{field}' treats '{name}' as an object, but it is a leaf field.")
            node = child if not seg.endswith("[]") else node["properties"][name]["items"]

        leaf_name = segments[-1]
        if leaf_name in node["properties"]:
            raise ValueError(f"Field '{field}' conflicts with an earlier definition of '{leaf_name}'.")
        node["properties"][leaf_name] = _leaf_schema(row_type, description)
        if required:
            node["required"].append(leaf_name)

    _strip_empty_required(schema)
    return schema


def _strip_empty_required(node: dict):
    if node.get("type") == "object":
        if not node.get("required"):
            node.pop("required", None)
        for child in node.get("properties", {}).values():
            _strip_empty_required(child)
    elif node.get("type") == "array":
        _strip_empty_required(node.get("items", {}))


_LEAF_JSON_TYPES = {"string", "number", "integer", "boolean"}


def schema_to_rows(schema: dict) -> List[dict]:
    """Inverse of rows_to_schema. Raises ValueError on constructs the builder can't represent."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Schema root must be an object.")
    rows = []
    _walk_object(schema, "", rows)
    if not rows:
        raise ValueError("Schema has no fields.")
    return rows


_UNSUPPORTED_KEYS = ("enum", "anyOf", "oneOf", "allOf", "$ref", "additionalProperties", "patternProperties")


def _check_supported(node: dict, path: str):
    for key in _UNSUPPORTED_KEYS:
        if node.get(key):
            raise ValueError(f"'{key}' at '{path or '(root)'}' is not supported by the builder.")


def _walk_object(node: dict, prefix: str, rows: List[dict]):
    _check_supported(node, prefix)
    required = set(node.get("required", []))
    for name, child in node.get("properties", {}).items():
        path = f"{prefix}.{name}" if prefix else name
        if not _SEGMENT_RE.match(name):
            raise ValueError(f"Field name '{name}' is not representable in the builder.")
        _check_supported(child, path)
        ctype = child.get("type")
        if ctype == "object":
            _walk_object(child, path, rows)
        elif ctype == "array":
            items = child.get("items", {})
            _check_supported(items, path)
            itype = items.get("type")
            if itype == "object":
                _walk_object(items, f"{path}[]", rows)
            elif itype in _LEAF_JSON_TYPES:
                rows.append(_row(path, f"{itype}[]", child, name in required))
            else:
                raise ValueError(f"Array items at '{path}' must be objects or simple types.")
        elif ctype in _LEAF_JSON_TYPES:
            rows.append(_row(path, ctype, child, name in required))
        else:
            raise ValueError(f"Type '{ctype}' at '{path}' is not supported by the builder.")


def _row(field: str, row_type: str, node: dict, required: bool) -> dict:
    return {
        "field": field,
        "type": row_type,
        "description": node.get("description", ""),
        "required": required,
    }


def validate_schema(schema: dict) -> str | None:
    """Validate the schema the same way the vLLM path does. Returns an error message or None."""
    try:
        create_model(schema)
    except Exception as e:
        return str(e)
    return None


# Schema library: one pretty-printed JSON file per named schema.

def _schema_path(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not safe:
        raise ValueError(f"Invalid schema name: '{name}'")
    return os.path.join(settings.SCHEMA_DIR, f"{safe}.json")


def list_schemas() -> List[str]:
    if not os.path.isdir(settings.SCHEMA_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(settings.SCHEMA_DIR)
        if f.endswith(".json")
    )


def load_schema(name: str) -> dict:
    with open(_schema_path(name)) as f:
        return json.load(f)


def save_schema(name: str, schema: dict):
    os.makedirs(settings.SCHEMA_DIR, exist_ok=True)
    with open(_schema_path(name), "w") as f:
        json.dump(schema, f, indent=2)


def delete_schema(name: str):
    os.remove(_schema_path(name))
