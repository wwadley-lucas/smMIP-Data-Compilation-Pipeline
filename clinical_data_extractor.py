#!/usr/bin/env python3
"""
clinical_data_extractor.py

Parse order sheets and extract clinical metadata for Master_Output.
Handles age formula evaluation and sample ID normalization.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# Clinical data column names as they appear in order sheet headers.
# Mapping: standardized internal name → expected header name in the Excel file.
# Falls back to positional extraction (Q-Y, 0-indexed 16-24) if name-based
# lookup fails, for backwards compatibility with header-less sheets.
CLINICAL_COLUMN_NAMES = {
    'D#': 'D#',
    'UCI_ID': 'UCI_ID',
    'Date': 'Date',
    'Age': 'Age',
    'Sex': 'Sex',
    'MPN': 'MPN',
    'Driver_Mutant': 'Driver_Mutant',
    'Medication': 'Medication',
    'IFNa': 'IFNa',
}

# Positional fallback indices (0-indexed column positions Q-Y)
CLINICAL_COLUMNS_POSITIONAL = {
    'D#': 16,           # Column Q - Patient ID
    'UCI_ID': 17,       # Column R - UCI identifier
    'Date': 18,         # Column S - Collection date
    'Age': 19,          # Column T - Age at collection
    'Sex': 20,          # Column U - Sex
    'MPN': 21,          # Column V - MPN status
    'Driver_Mutant': 22, # Column W - Driver mutation
    'Medication': 23,    # Column X - Medication
    'IFNa': 24          # Column Y - Interferon alpha
}


def evaluate_age_formula(value) -> Optional[int]:
    """
    Evaluate age formulas like '=2019-1942' → 77.
    Also handles plain numeric values.

    Args:
        value: Cell value (may be string formula, int, or float)

    Returns:
        Age as integer, or None if cannot be parsed
    """
    if pd.isna(value):
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        return int(value) if not pd.isna(value) else None

    # String value
    value_str = str(value).strip()

    # Handle Excel formula format: =YEAR-YEAR
    # LIMITATION: Year subtraction (e.g., =2019-1942) gives an approximate age
    # that can be off by +/-1 year since it ignores month/day of birth and
    # collection date. This matches how the order sheets encode age.
    if value_str.startswith('='):
        formula = value_str[1:]
        # Match pattern like "2019-1942"
        match = re.match(r'(\d{4})\s*-\s*(\d{4})', formula)
        if match:
            year1, year2 = int(match.group(1)), int(match.group(2))
            return abs(year1 - year2)

    # Try direct numeric conversion
    try:
        return int(float(value_str))
    except (ValueError, TypeError):
        return None


def normalize_sample_id(sample_id: str) -> str:
    """
    Normalize sample ID from mutation file format to order sheet format.

    Examples:
        "D79A,D79B" → "D79"
        "D79A" → "D79"
        "D79" → "D79"

    Args:
        sample_id: Raw sample ID from called_mutations.txt

    Returns:
        Normalized sample ID matching order sheet D# format
    """
    if pd.isna(sample_id):
        return ""

    sample_id = str(sample_id).strip()

    # Take first part if comma-separated
    base_id = sample_id.split(',')[0].strip()

    # Remove trailing A or B suffix only from D-number IDs (e.g., D79A → D79)
    # to avoid stripping meaningful trailing letters from non-D-number identifiers
    normalized = re.sub(r'^(D\d+)[AB]$', r'\1', base_id)

    return normalized


def extract_clinical_from_order_sheet(
    order_sheet_path: str,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract clinical data from an order sheet Excel file.

    Args:
        order_sheet_path: Path to the order sheet Excel file
        sheet_name: Specific sheet name, or None for first sheet

    Returns:
        DataFrame with columns: D#, UCI_ID, Date, Age, Sex, MPN, Driver_Mutant, Medication, IFNa
    """
    try:
        # Read Excel file
        if sheet_name:
            df = pd.read_excel(order_sheet_path, sheet_name=sheet_name, header=0)
        else:
            df = pd.read_excel(order_sheet_path, header=0)

        # Try name-based extraction first (preferred — resilient to column reordering)
        header_names = set(df.columns.astype(str))
        expected_names = set(CLINICAL_COLUMN_NAMES.values())
        matched_names = header_names & expected_names

        if len(matched_names) >= 3:
            # Enough named columns found — use name-based extraction
            logger.debug(f"Using name-based column extraction ({len(matched_names)} columns matched)")
            clinical_data = {}
            for std_name, header_name in CLINICAL_COLUMN_NAMES.items():
                if header_name in df.columns:
                    clinical_data[std_name] = df[header_name]
                else:
                    clinical_data[std_name] = None
        else:
            # Fall back to positional extraction for header-less or non-standard sheets
            logger.debug("Falling back to positional column extraction")
            if len(df.columns) < 25:
                logger.warning(f"Order sheet has only {len(df.columns)} columns, expected at least 25")
                return pd.DataFrame()

            clinical_data = {}
            for col_name, col_idx in CLINICAL_COLUMNS_POSITIONAL.items():
                if col_idx < len(df.columns):
                    clinical_data[col_name] = df.iloc[:, col_idx]
                else:
                    clinical_data[col_name] = None

        result_df = pd.DataFrame(clinical_data)

        # Process Age column - evaluate formulas
        if 'Age' in result_df.columns:
            result_df['Age'] = result_df['Age'].apply(evaluate_age_formula)

        # Clean D# column - ensure string type
        if 'D#' in result_df.columns:
            result_df['D#'] = result_df['D#'].astype(str).str.strip()
            # Remove rows where D# is empty or 'nan'
            result_df = result_df[
                (result_df['D#'].notna()) &
                (result_df['D#'] != '') &
                (result_df['D#'].str.lower() != 'nan')
            ]

        # Parse Date column
        if 'Date' in result_df.columns:
            result_df['Date'] = pd.to_datetime(result_df['Date'], errors='coerce')

        return result_df

    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        logger.error(f"Error reading order sheet {order_sheet_path}: {type(e).__name__}: {e}")
        return pd.DataFrame()


def extract_clinical_from_metadata_txt(metadata_path: str) -> pd.DataFrame:
    """
    Parse MetaData.txt format (tab-delimited).

    Expected columns: Sample_ID, UCI_ID, Date_collected, Age_collected, Sex,
                      MPN, Driver_Mut, Family-History, Medication, IFNA, ASXL1, TET2

    Args:
        metadata_path: Path to MetaData.txt file

    Returns:
        DataFrame with standardized clinical columns
    """
    try:
        df = pd.read_csv(metadata_path, sep='\t')

        # Rename columns to standard format
        column_map = {
            'Sample_ID': 'D#',
            'UCI_ID': 'UCI_ID',
            'Date_collected': 'Date',
            'Age_collected': 'Age',
            'Sex': 'Sex',
            'MPN': 'MPN',
            'Driver_Mut': 'Driver_Mutant',
            'Family-History': 'Family_History',
            'Medication': 'Medication',
            'IFNA': 'IFNa'
        }

        # Rename existing columns
        df = df.rename(columns=column_map)

        # Select standard columns that exist
        standard_cols = ['D#', 'UCI_ID', 'Date', 'Age', 'Sex', 'MPN', 'Driver_Mutant', 'Medication', 'IFNa']
        available_cols = [c for c in standard_cols if c in df.columns]
        result_df = df[available_cols].copy()

        # Clean D# column - ensure string type
        if 'D#' in result_df.columns:
            result_df['D#'] = result_df['D#'].astype(str).str.strip()
            result_df = result_df[
                (result_df['D#'].notna()) &
                (result_df['D#'] != '') &
                (result_df['D#'].str.lower() != 'nan')
            ]

        # Process Age column - convert to numeric
        if 'Age' in result_df.columns:
            result_df['Age'] = pd.to_numeric(result_df['Age'], errors='coerce')

        # Standardize Sex column - uppercase
        if 'Sex' in result_df.columns:
            result_df['Sex'] = result_df['Sex'].str.upper()

        # Standardize MPN column - uppercase
        if 'MPN' in result_df.columns:
            result_df['MPN'] = result_df['MPN'].str.upper()

        logger.info(f"Extracted {len(result_df)} records from MetaData.txt")
        return result_df

    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        logger.error(f"Error reading MetaData.txt {metadata_path}: {type(e).__name__}: {e}")
        return pd.DataFrame()


def find_clinical_data(batch_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find clinical data file in a batch directory.

    Searches for MetaData.txt first (preferred), then order sheets.

    Args:
        batch_dir: Path to batch directory

    Returns:
        Tuple of (path, format_type) where format_type is 'metadata_txt' or 'order_sheet'
        Returns (None, None) if not found
    """
    batch_path = Path(batch_dir)

    # Check for MetaData.txt first (case-insensitive patterns)
    metadata_patterns = ['MetaData.txt', 'metadata.txt', 'Metadata.txt', 'METADATA.txt']

    # Search in batch directory and FINAL_OUT
    search_dirs = [batch_path, batch_path / 'FINAL_OUT']

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in metadata_patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                return str(matches[0]), 'metadata_txt'

    # Fall back to order sheet search
    order_sheet = find_order_sheet(batch_dir)
    if order_sheet:
        return order_sheet, 'order_sheet'

    return None, None


def find_order_sheet(batch_dir: str) -> Optional[str]:
    """
    Find the order sheet file in a batch directory.

    Looks for files matching patterns like:
    - smMIP Order Sheet_*.xlsx
    - *Order*Sheet*.xlsx
    - order_sheet*.xlsx

    Args:
        batch_dir: Path to batch directory

    Returns:
        Path to order sheet file, or None if not found
    """
    batch_path = Path(batch_dir)

    # Common order sheet patterns
    patterns = [
        'smMIP Order Sheet*.xlsx',
        '*Order*Sheet*.xlsx',
        '*order*sheet*.xlsx',
        'Order*.xlsx'
    ]

    for pattern in patterns:
        matches = list(batch_path.glob(pattern))
        if matches:
            # Return first match (most recent if multiple)
            return str(sorted(matches, key=os.path.getmtime, reverse=True)[0])

    return None


def build_clinical_linkage(
    batches: Dict[str, str],
    output_path: str,
    external_metadata: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Build clinical linkage table from all batch clinical data files.

    Supports both order sheet (Excel) and MetaData.txt formats.

    Args:
        batches: Dict mapping batch_id to batch directory path
        output_path: Path to save clinical_linkage.csv
        external_metadata: Dict mapping batch_id to external metadata file path

    Returns:
        DataFrame with clinical data from all batches
    """
    all_clinical = []
    external_metadata = external_metadata or {}

    for batch_id, batch_dir in batches.items():
        # Check for external metadata override first
        if batch_id in external_metadata:
            ext_path = external_metadata[batch_id]
            if os.path.exists(ext_path):
                logger.info(f"Using external metadata for {batch_id}: {ext_path}")
                if ext_path.lower().endswith('.txt'):
                    clinical_df = extract_clinical_from_metadata_txt(ext_path)
                else:
                    clinical_df = extract_clinical_from_order_sheet(ext_path)

                if not clinical_df.empty:
                    clinical_df['Batch_ID'] = batch_id
                    all_clinical.append(clinical_df)
                continue

        # Find clinical data in batch directory
        clinical_path, format_type = find_clinical_data(batch_dir)

        if clinical_path:
            logger.info(f"Extracting clinical data from {batch_id} ({format_type}): {clinical_path}")

            if format_type == 'metadata_txt':
                clinical_df = extract_clinical_from_metadata_txt(clinical_path)
            else:
                clinical_df = extract_clinical_from_order_sheet(clinical_path)

            if not clinical_df.empty:
                clinical_df['Batch_ID'] = batch_id
                all_clinical.append(clinical_df)
        else:
            logger.warning(f"No clinical data found for batch {batch_id}")

    if all_clinical:
        combined = pd.concat(all_clinical, ignore_index=True)

        # Save to CSV
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved clinical linkage to {output_path}")

        return combined

    return pd.DataFrame()


def link_mutation_to_clinical(
    sample_id: str,
    clinical_df: pd.DataFrame
) -> Optional[Dict]:
    """
    Link a mutation sample ID to clinical data.

    Args:
        sample_id: Sample ID from called_mutations.txt (e.g., "D79A,D79B")
        clinical_df: Clinical linkage DataFrame

    Returns:
        Dict with clinical data, or None if not found
    """
    normalized_id = normalize_sample_id(sample_id)

    if clinical_df.empty or 'D#' not in clinical_df.columns:
        return None

    # Find matching row
    matches = clinical_df[clinical_df['D#'] == normalized_id]

    if not matches.empty:
        return matches.iloc[0].to_dict()

    return None


if __name__ == '__main__':
    # Test with KG001 order sheet
    import sys

    logging.basicConfig(level=logging.INFO)

    smmip_root = os.environ.get("SMMIP_ROOT", ".")
    test_path = os.path.join(smmip_root, "KG001_01.22.25", "smMIP Order Sheet_KG001.xlsx")

    if os.path.exists(test_path):
        print(f"Testing clinical extraction from: {test_path}")
        df = extract_clinical_from_order_sheet(test_path)
        print(f"\nExtracted {len(df)} rows:")
        print(df.head(10))

        # Test sample ID normalization
        test_ids = ["D79A,D79B", "D98A", "D100", "D101B"]
        print("\nSample ID normalization:")
        for sid in test_ids:
            print(f"  {sid} → {normalize_sample_id(sid)}")
    else:
        print(f"Test file not found: {test_path}")
        print("Set SMMIP_ROOT to the smMIP data directory.")
        sys.exit(1)
