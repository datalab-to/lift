import json
from pathlib import Path
from typing import List

import click

from lift.extract import extract_images, resolve_schema
from lift.input import load_file
from lift.model import InferenceManager


def get_supported_files(input_path: Path) -> List[Path]:
    """Get list of supported image/PDF files from path."""
    supported_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".tiff",
        ".bmp",
    }

    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        else:
            raise click.BadParameter(f"Unsupported file type: {input_path.suffix}")

    elif input_path.is_dir():
        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(files)

    else:
        raise click.BadParameter(f"Path does not exist: {input_path}")


def save_output(output_dir: Path, file_name: str, result, num_pages: int):
    """Save the extraction and metadata for one file to the output directory."""
    safe_name = Path(file_name).stem

    if result.extraction is not None:
        extraction_path = output_dir / f"{safe_name}.json"
        with open(extraction_path, "w", encoding="utf-8") as f:
            json.dump(result.extraction, f, indent=2)
        click.echo(f"  Saved: {extraction_path}")

    metadata = {
        "file_name": file_name,
        "num_pages": num_pages,
        "token_count": result.token_count,
        "error": result.error,
    }
    if result.extraction is None:
        # Keep the raw model output around so failures can be debugged
        metadata["raw"] = result.raw

    metadata_path = output_dir / f"{safe_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--schema",
    "schema_arg",
    required=True,
    help="Path to a JSON schema file, an inline JSON schema string, or the name of a saved schema in the schema library.",
)
@click.option(
    "--method",
    type=click.Choice(["hf", "vllm"], case_sensitive=False),
    default="vllm",
    help="Inference method: 'hf' for local model, 'vllm' for vLLM server.",
)
@click.option(
    "--page-range",
    type=str,
    default=None,
    help="Page range for PDFs (e.g., '0-5,7,9-12'). Only applicable to PDF files.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Maximum number of output tokens.",
)
def main(
    input_path: Path,
    output_path: Path,
    schema_arg: str,
    method: str,
    page_range: str,
    max_output_tokens: int,
):
    """Extract structured data matching a JSON schema from PDFs and images."""
    # Resolve the schema up front so bad input fails before the model loads
    try:
        schema = resolve_schema(schema_arg)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--schema")

    click.echo("lift CLI - Starting extraction")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Method: {method}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    click.echo(f"\nLoading model with method '{method}'...")
    model = InferenceManager(method=method)
    click.echo("Model loaded successfully.")

    # Get files to process
    files_to_process = get_supported_files(input_path)
    click.echo(f"\nFound {len(files_to_process)} file(s) to process.")

    if not files_to_process:
        click.echo("No supported files found. Exiting.")
        return

    # Process each file
    for file_idx, file_path in enumerate(files_to_process, 1):
        click.echo(
            f"\n[{file_idx}/{len(files_to_process)}] Processing: {file_path.name}"
        )

        try:
            # Load images from file
            config = {"page_range": page_range} if page_range else {}
            images = load_file(str(file_path), config)
            click.echo(f"  Loaded {len(images)} page(s)")

            # Run extraction on all pages at once
            result = extract_images(
                images, schema, model, max_output_tokens=max_output_tokens
            )

            if result.error or result.extraction is None:
                click.echo(
                    f"  Extraction failed for {file_path.name}. Raw output saved to metadata.",
                    err=True,
                )

            save_output(output_path, file_path.name, result, num_pages=len(images))
            click.echo(f"  Completed: {file_path.name}")

        except Exception as e:
            click.echo(f"  Error processing {file_path.name}: {e}", err=True)
            continue

    click.echo(f"\nProcessing complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
