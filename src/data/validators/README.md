# Data Validators

This module provides data validation utilities for the ML CI/CD pipeline.

## Features

- **Schema Validation**: Validates dataframes against required column schemas
- **Data Quality Checks**: Detects missing values and data quality issues
- **CLI Interface**: Command-line tool for CI/CD integration
- **Programmatic API**: Python API for use in scripts and notebooks

## Required Schema

All datasets must contain the following columns:

- `sepal_length`: Sepal length measurement
- `sepal_width`: Sepal width measurement
- `petal_length`: Petal length measurement
- `petal_width`: Petal width measurement
- `target`: Target class label

## CLI Usage

### Validate Sample Dataset

```bash
python -m src.data.validators.cli --sample
```

### Validate Specific CSV Files

```bash
python -m src.data.validators.cli data/processed/train.csv data/processed/test.csv
```

### Validate All CSV Files in Directory

```bash
python -m src.data.validators.cli data/processed/*.csv
```

## Programmatic API

### Basic Usage

```python
from src.data import validators

# Load and validate sample data
df = validators.load_sample_dataframe()
is_valid, message = validators.validate_dataframe(df)

if is_valid:
    print(f"Validation passed: {message}")
else:
    print(f"Validation failed: {message}")
```

### Validate CSV Files

```python
from pathlib import Path
from src.data.validators import validate_csv_file

file_path = Path("data/processed/train.csv")
is_valid, message = validate_csv_file(file_path)
```

### Access Required Columns

```python
from src.data.validators import REQUIRED_COLUMNS

print(f"Required columns: {REQUIRED_COLUMNS}")
```

## Validation Rules

1. **Required Columns**: All columns in `REQUIRED_COLUMNS` must be present
2. **No Missing Values**: No NaN values allowed in required columns
3. **Valid CSV Format**: Files must be readable as CSV with proper formatting

## CI/CD Integration

The data validation workflow (`.github/workflows/data-validation.yml`) automatically runs validation on:

- Push events that modify files in `src/data/**` or `data/**`
- Manual workflow dispatch

## Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

## Error Messages

- `"Missing required columns: [...]"`: Dataset is missing one or more required columns
- `"NaN values present in required columns"`: Dataset contains missing values
- `"File not found: ..."`: Specified file does not exist
- `"Empty file: ..."`: CSV file is empty
- `"Error reading file: ..."`: Generic error reading or parsing CSV

## Testing

Run the test suite:

```bash
pytest tests/unit/test_data_validators.py -v
```

## Architecture

```
src/data/validators/
├── __init__.py      # Core validation functions
├── cli.py          # Command-line interface
├── __main__.py     # Module entry point
└── README.md       # This file
```
