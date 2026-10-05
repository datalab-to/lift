"""Optional document-authenticity verification via the Stipple API.

Stipple (https://www.stipple.sh) inspects a document for forensic
authenticity signals — a risk band, inspection quality, and per-signal
evidence — and detects AI-written prose. Pairing it with lift adds a trust
signal to extracted JSON: the fields are what the document *shows*; the
warrant says something about whether the document itself is trustworthy.

Free anonymous tier: no API key required. Set STIPPLE_API_KEY for your own
metering. Every function is best-effort: failures return None / a warning
entry and never break an extraction run.
"""
import base64
import json
import os
from pathlib import Path
from typing import Optional

import click

STIPPLE_BASE_URL = "https://www.stipple.sh"
_REQUEST_TIMEOUT = 300  # seconds


def _headers() -> dict:
    headers = {
        "User-Agent": "lift-stipple-verification/1.0",
        "Accept": "application/json",
    }
    api_key = os.environ.get("STIPPLE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _post_file(endpoint: str, file_path: Path) -> Optional[dict]:
    """POST a file as multipart/form-data to a Stipple endpoint."""
    try:
        import urllib.request
        import uuid

        boundary = "----lift-stipple" + uuid.uuid4().hex
        with open(file_path, "rb") as f:
            content = f.read()
        body = b"".join(
            [
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{file_path.name}"\r\n'
                    f"Content-Type: application/octet-stream\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            STIPPLE_BASE_URL + endpoint,
            data=body,
            method="POST",
            headers={
                **_headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001 - verification is best-effort
        click.echo(f"  Verification unavailable: {e}", err=True)
        return None


def verify_document(file_path: Path) -> Optional[dict]:
    """Forensic authenticity inspection. Returns a compact warrant or None."""
    result = _post_file("/v1/warrants", file_path)
    if not result:
        return None
    return {
        "warrant_id": result.get("warrant_id"),
        "risk_band": result.get("risk_band"),
        "risk_score": result.get("risk_score"),
        "inspection_quality": result.get("inspection_quality"),
        "recommended_action": result.get("recommended_action"),
        "summary": result.get("summary"),
    }


def detect_ai_text(file_path: Path) -> Optional[dict]:
    """AI-written-prose probability for the document's text. None on failure."""
    result = _post_file("/v1/detect-ai-text", file_path)
    if not result:
        return None
    if result.get("applicable") is False:
        # Non-prose documents (forms, tables) are abstained on server-side.
        return {"applicable": False}
    return {
        "applicable": True,
        "probability": result.get("probability"),
        "lean": result.get("lean"),
        "tells": result.get("tells"),
    }


def verification_block(file_path: Path, enabled: bool) -> Optional[dict]:
    """Build the document_verification metadata block for a file.

    Returns None when disabled or when both checks fail (offline etc.) —
    callers should simply omit the block in that case.
    """
    if not enabled:
        return None
    block = {}
    warrant = verify_document(file_path)
    if warrant:
        block["authenticity"] = warrant
    ai = detect_ai_text(file_path)
    if ai:
        block["ai_text"] = ai
    return block or None
