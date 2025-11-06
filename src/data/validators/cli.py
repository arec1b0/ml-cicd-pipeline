"""
CLI interface for data validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from . import validate_csv_file, validate_dataframe, load_sample_dataframe


def validate_sample() -> int:
    """Validates the built-in sample iris dataset.

    This function loads a sample DataFrame, validates its schema and content,
    and prints the result to the console.

    Returns:
        An exit code, 0 for success and 1 for failure.
    """
    print("Validating sample dataset...")
    df = load_sample_dataframe()
    is_valid, message = validate_dataframe(df)

    if is_valid:
        print(f"✓ Sample validation passed: {message}")
        print(f"  - Validated {len(df)} rows")
        print(f"  - Schema: {list(df.columns)}")
        return 0
    else:
        print(f"✗ Sample validation failed: {message}")
        return 1


def validate_files(file_paths: List[Path]) -> int:
    """
    Validate one or more CSV files.

    Args:
        file_paths: List of CSV file paths to validate

    Returns:
        0 if all validations pass, 1 otherwise
    """
    all_valid = True

    for file_path in file_paths:
        print(f"\nValidating {file_path}...")
        is_valid, message = validate_csv_file(file_path)

        if is_valid:
            print(f"✓ {file_path.name}: {message}")
        else:
            print(f"✗ {file_path.name}: {message}")
            all_valid = False

    return 0 if all_valid else 1


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Data validation CLI for ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate sample dataset
  python -m src.data.validators.cli --sample

  # Validate specific CSV files
  python -m src.data.validators.cli data/processed/train.csv data/processed/test.csv

  # Validate all CSV files in a directory
  python -m src.data.validators.cli data/processed/*.csv
        """,
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Validate the sample iris dataset",
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="CSV files to validate",
    )

    args = parser.parse_args()

    if args.sample:
        return validate_sample()
    elif args.files:
        return validate_files(args.files)
    else:
        parser.print_help()
        print("\nError: Must specify either --sample or provide file paths")
        return 1


if __name__ == "__main__":
    sys.exit(main())
