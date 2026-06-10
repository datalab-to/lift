"""Extract structured data from a document via the Datalab API.

Submits a file + JSON schema to https://www.datalab.to/api/v1/extract and
polls until complete. Prints the extraction JSON to stdout; progress and
confidence info go to stderr.

Requires the DATALAB_API_KEY environment variable (get a key at
https://www.datalab.to). Uses httpx, which ships with lift (via openai).

Usage:
    python datalab_extract.py document.pdf --schema schema.json
    python datalab_extract.py doc.pdf --schema '{"type": "object", ...}' --mode balanced
    python datalab_extract.py doc.pdf --schema schema.json --page-range 0-5,8 > out.json
"""

import argparse
import json
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("DATALAB_BASE_URL", "https://www.datalab.to")


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_schema(schema_arg: str) -> str:
    """Accept a path to a JSON file or an inline JSON string; return JSON text."""
    if os.path.isfile(schema_arg):
        with open(schema_arg) as f:
            text = f.read()
    else:
        text = schema_arg
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        fail(f"schema is not valid JSON: {e}")
    if not isinstance(parsed, dict) or "properties" not in parsed:
        fail("schema must be a JSON object with a 'properties' key")
    return json.dumps(parsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured data from a document via the Datalab API."
    )
    parser.add_argument("file", help="PDF or image file to extract from")
    parser.add_argument(
        "--schema",
        required=True,
        help="JSON schema: path to a .json file or an inline JSON string",
    )
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["turbo", "fast", "balanced"],
        help="Extraction mode: turbo (fastest, image-only), fast (low latency), "
        "balanced (highest accuracy, per-field verification). Default: balanced.",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Pages to process, e.g. '0,2-4,10'",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Max seconds to wait for completion (default 600)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3,
        help="Seconds between status polls (default 3)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DATALAB_API_KEY")
    if not api_key:
        fail(
            "DATALAB_API_KEY is not set. Get an API key at https://www.datalab.to "
            "and export DATALAB_API_KEY=<key>."
        )
    if not os.path.isfile(args.file):
        fail(f"file not found: {args.file}")

    schema_json = resolve_schema(args.schema)
    headers = {"X-API-Key": api_key}

    # Submit.
    data = {
        "page_schema": schema_json,
        "extraction_mode": args.mode,
        "output_format": "json",
    }
    if args.page_range:
        data["page_range"] = args.page_range

    with open(args.file, "rb") as f:
        files = {"file": (os.path.basename(args.file), f)}
        resp = httpx.post(
            f"{BASE_URL}/api/v1/extract",
            headers=headers,
            files=files,
            data=data,
            timeout=60,
        )

    if resp.status_code != 200:
        fail(f"submit failed ({resp.status_code}): {resp.text[:500]}")
    body = resp.json()
    request_id = body.get("request_id")
    if not request_id:
        fail(f"no request_id in response: {body}")
    print(f"submitted: {request_id} (mode={args.mode})", file=sys.stderr)

    # Poll.
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        resp = httpx.get(
            f"{BASE_URL}/api/v1/extract/{request_id}", headers=headers, timeout=30
        )
        if resp.status_code != 200:
            fail(f"poll failed ({resp.status_code}): {resp.text[:300]}")
        body = resp.json()
        status = body.get("status")

        if status == "complete":
            if body.get("success") is False or body.get("error"):
                fail(f"extraction failed: {body.get('error', 'unknown error')}")
            extraction = body.get("extraction_schema_json")
            if extraction is None:
                fail("complete but no extraction_schema_json in response")
            if isinstance(extraction, str):
                extraction = json.loads(extraction)
            score = body.get("extraction_score_average")
            if score is not None:
                print(f"confidence score: {score:.2f}/5", file=sys.stderr)
            print(json.dumps(extraction, indent=2))
            return

        time.sleep(args.poll_interval)

    fail(f"timed out after {args.timeout}s (request_id={request_id})")


if __name__ == "__main__":
    main()
