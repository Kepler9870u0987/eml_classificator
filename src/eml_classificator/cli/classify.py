"""
Command-line interface for email classification.

Provides CLI for batch processing .eml files with model configuration.

Usage:
    # Single file
    python -m eml_classificator.cli.classify input.eml

    # With model override
    python -m eml_classificator.cli.classify input.eml --model ollama/deepseek-r1:7b

    # Directory batch processing
    python -m eml_classificator.cli.classify emails/ --output results.jsonl

    # With environment variable
    export LLM_MODEL="ollama/qwen2.5:32b"
    python -m eml_classificator.cli.classify input.eml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import structlog

from eml_classificator.parsing.eml_parser import parse_eml_bytes
from eml_classificator.models.email_document import build_email_document
from eml_classificator.candidates import extract_candidates
from eml_classificator.classification.classifier import classify_email
from eml_classificator.config import settings
from eml_classificator.logging_config import setup_logging


# Setup logging
setup_logging()
logger = structlog.get_logger(__name__)


# ============================================================================
# CLI FUNCTIONS
# ============================================================================

def process_single_file(
    eml_path: Path,
    model_override: Optional[str] = None,
    dictionary_version: int = 1,
    verbose: bool = False
) -> dict:
    """
    Process a single .eml file through the complete pipeline.

    Args:
        eml_path: Path to .eml file
        model_override: Optional LLM model override
        dictionary_version: Dictionary version
        verbose: Enable verbose output

    Returns:
        Classification result as dict

    Raises:
        Exception: On processing errors
    """
    if verbose:
        logger.info("processing_file", path=str(eml_path))

    # Phase 1: Parse
    with open(eml_path, 'rb') as f:
        eml_bytes = f.read()

    from eml_classificator.parsing.eml_parser import parse_eml_bytes
    msg = parse_eml_bytes(eml_bytes)
    email_document = build_email_document(msg)

    if verbose:
        logger.info(
            "phase1_completed",
            message_id=email_document.message_id,
            subject=email_document.subject[:50]
        )

    # Phase 2: Candidates
    candidates_result = extract_candidates(email_document)

    if verbose:
        logger.info(
            "phase2_completed",
            candidates_count=len(candidates_result.rich_candidates)
        )

    # Phase 3: Classify
    classification_result = classify_email(
        email_document=email_document.model_dump(),
        candidates=[c.model_dump() for c in candidates_result.rich_candidates],
        dictionary_version=dictionary_version,
        model_override=model_override
    )

    if verbose:
        logger.info(
            "phase3_completed",
            topics=[t.label_id for t in classification_result.topics],
            sentiment=classification_result.sentiment.value,
            priority=classification_result.priority.value
        )

    return classification_result.model_dump()


def process_directory(
    dir_path: Path,
    model_override: Optional[str] = None,
    dictionary_version: int = 1,
    batch_size: int = 10,
    verbose: bool = False
) -> List[dict]:
    """
    Process all .eml files in a directory.

    Args:
        dir_path: Directory path
        model_override: Optional LLM model override
        dictionary_version: Dictionary version
        batch_size: Batch size for processing
        verbose: Enable verbose output

    Returns:
        List of classification results
    """
    eml_files = list(dir_path.glob("**/*.eml"))

    if not eml_files:
        logger.warning("no_eml_files_found", directory=str(dir_path))
        return []

    logger.info("processing_directory", files_count=len(eml_files))

    results = []
    errors = []

    for idx, eml_file in enumerate(eml_files, 1):
        try:
            if verbose:
                print(f"[{idx}/{len(eml_files)}] Processing {eml_file.name}...")

            result = process_single_file(
                eml_path=eml_file,
                model_override=model_override,
                dictionary_version=dictionary_version,
                verbose=verbose
            )

            results.append(result)

        except Exception as e:
            logger.error("file_processing_failed", file=str(eml_file), error=str(e))
            errors.append({
                "file": str(eml_file),
                "error": str(e)
            })

    logger.info(
        "directory_processing_completed",
        total=len(eml_files),
        success=len(results),
        errors=len(errors)
    )

    return results


def write_output(results: List[dict], output_path: Optional[Path], format: str = "jsonl"):
    """
    Write results to file.

    Args:
        results: List of classification results
        output_path: Output file path
        format: Output format ("json" or "jsonl")
    """
    if not output_path:
        # Print to stdout
        if format == "jsonl":
            for result in results:
                print(json.dumps(result, ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        if format == "jsonl":
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("output_written", path=str(output_path), count=len(results))


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Email Classification CLI - Process .eml files through complete pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with default model
  %(prog)s input.eml

  # Override model
  %(prog)s input.eml --model ollama/deepseek-r1:7b

  # Process directory, save to file
  %(prog)s emails/ --output results.jsonl

  # Use environment variable for model
  export LLM_MODEL="ollama/qwen2.5:32b"
  %(prog)s input.eml

  # Verbose output
  %(prog)s input.eml --verbose

Supported models:
  - ollama/<model>:<tag>  (e.g., ollama/deepseek-r1:7b, ollama/qwen2.5:32b)
  - openai/<model>        (e.g., openai/gpt-4o) - requires OPENAI_API_KEY
  - deepseek/<model>      (e.g., deepseek/deepseek-chat) - requires DEEPSEEK_API_KEY
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to .eml file or directory containing .eml files"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Override LLM model (format: provider/model-name, e.g., 'ollama/deepseek-r1:7b')"
    )

    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="Override LLM provider only (ollama, openai, deepseek, openrouter)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout). Format auto-detected from extension (.json or .jsonl)"
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format (default: jsonl)"
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Batch size for directory processing (default: 10)"
    )

    parser.add_argument(
        "--dictionary-version",
        "-d",
        type=int,
        default=1,
        help="Dictionary version for audit trail (default: 1)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to custom config file (not yet implemented)"
    )

    args = parser.parse_args()

    # Setup model override
    model_override = args.model

    # If provider specified without model, construct model string
    if args.provider and not model_override:
        model_override = f"{args.provider}/{settings.llm_model}"

    # Check if input is file or directory
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Process
        if input_path.is_file():
            result = process_single_file(
                eml_path=input_path,
                model_override=model_override,
                dictionary_version=args.dictionary_version,
                verbose=args.verbose
            )
            results = [result]

        elif input_path.is_dir():
            results = process_directory(
                dir_path=input_path,
                model_override=model_override,
                dictionary_version=args.dictionary_version,
                batch_size=args.batch_size,
                verbose=args.verbose
            )

        else:
            print(f"Error: Invalid input path: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Write output
        output_path = Path(args.output) if args.output else None

        # Auto-detect format from file extension
        if output_path and args.format == "jsonl":
            if output_path.suffix == ".json":
                format = "json"
            else:
                format = "jsonl"
        else:
            format = args.format

        write_output(results, output_path, format)

        if args.verbose:
            print(f"\n✓ Processed {len(results)} emails successfully", file=sys.stderr)

    except Exception as e:
        logger.error("cli_failed", error=str(e), exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
