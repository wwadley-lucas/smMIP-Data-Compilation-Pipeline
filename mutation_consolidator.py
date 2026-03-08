#!/usr/bin/env python3
"""
mutation_consolidator.py

Merge all batch called_mutations.txt files into a master consolidated file.
Adds batch_id column and prevents duplicate processing via manifest tracking.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Expected columns in called_mutations.txt (29 columns)
MUTATION_COLUMNS = [
    'sample_ID', 'smMIP', 'chr', 'pos', 'ref', 'alt', 'gene', 'protein',
    'cosmic', 'maf', 'variant_type', 'cadd_scaled', 'P-value',
    'P-value.Bonferroni', 'num.pval.pass', 'pass.pval.in.pairs',
    'non.ref.counts', 'total.depth', 'allele.frequency',
    'samples.with.higher.vaf', 'higher.vaf', 'lower.vaf',
    'SSCS.non.ref.counts', 'SSCS.total.depth', 'SSCS.family.size',
    'SSCS.in.family.non.ref.vaf', 'SSCS.allele.frequency', 'SSCS.overlap', 'flags'
]


def find_mutations_file(batch_dir: str) -> Optional[str]:
    """
    Find called_mutations.txt in a batch directory.

    Searches in:
    1. FINAL_OUT/called_mutations.txt
    2. batch_dir/called_mutations.txt

    Args:
        batch_dir: Path to batch directory

    Returns:
        Path to mutations file, or None if not found
    """
    batch_path = Path(batch_dir)

    # Check FINAL_OUT first
    final_out = batch_path / 'FINAL_OUT' / 'called_mutations.txt'
    if final_out.exists():
        return str(final_out)

    # Check batch root
    root_file = batch_path / 'called_mutations.txt'
    if root_file.exists():
        return str(root_file)

    # Search recursively as fallback
    for f in batch_path.rglob('called_mutations.txt'):
        return str(f)

    return None


def read_mutations_file(
    file_path: str,
    batch_id: str
) -> pd.DataFrame:
    """
    Read a called_mutations.txt file and add batch_id column.

    Args:
        file_path: Path to called_mutations.txt
        batch_id: Batch identifier to add

    Returns:
        DataFrame with mutations data and batch_id column
    """
    try:
        # Read tab-delimited file
        df = pd.read_csv(file_path, sep='\t', dtype=str)

        # Verify expected columns are present (check key ones)
        required_cols = ['sample_ID', 'gene', 'variant_type', 'SSCS.allele.frequency']
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            logger.warning(f"Missing expected columns in {file_path}: {missing}")
            # Try to read with assumed column names
            df = pd.read_csv(file_path, sep='\t', names=MUTATION_COLUMNS, dtype=str)

        # Add batch_id as first column
        df.insert(0, 'batch_id', batch_id)

        # Convert numeric columns
        numeric_cols = ['pos', 'maf', 'cadd_scaled', 'P-value', 'P-value.Bonferroni',
                        'non.ref.counts', 'total.depth', 'allele.frequency',
                        'SSCS.non.ref.counts', 'SSCS.total.depth',
                        'SSCS.allele.frequency', 'SSCS.overlap']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Read {len(df)} mutations from {batch_id}")
        return df

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()


def consolidate_mutations(
    batches: dict,
    output_path: str,
    processed_batches: set = None,
    external_mutations: Optional[dict] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Consolidate mutations from multiple batches into a single file.

    Args:
        batches: Dict mapping batch_id to batch directory path
        output_path: Path to save master_called_mutations.txt
        processed_batches: Set of batch_ids already processed (to skip)
        external_mutations: Dict mapping batch_id to external mutations file path

    Returns:
        Tuple of (consolidated DataFrame, list of newly processed batch_ids)
    """
    if processed_batches is None:
        processed_batches = set()
    if external_mutations is None:
        external_mutations = {}

    all_mutations = []
    newly_processed = []

    for batch_id, batch_dir in batches.items():
        # Skip if already processed
        if batch_id in processed_batches:
            logger.info(f"Skipping already processed batch: {batch_id}")
            continue

        # Check for external override first
        if batch_id in external_mutations:
            ext_path = external_mutations[batch_id]
            if os.path.exists(ext_path):
                logger.info(f"Using external mutations file for {batch_id}: {ext_path}")
                df = read_mutations_file(ext_path, batch_id)
                if not df.empty:
                    all_mutations.append(df)
                    newly_processed.append(batch_id)
                continue

        # Fall back to standard location
        mutations_file = find_mutations_file(batch_dir)

        if mutations_file:
            df = read_mutations_file(mutations_file, batch_id)
            if not df.empty:
                all_mutations.append(df)
                newly_processed.append(batch_id)
        else:
            logger.warning(f"No called_mutations.txt found for batch {batch_id}")

    if all_mutations:
        # Combine all new mutations
        new_data = pd.concat(all_mutations, ignore_index=True)

        # If master file exists, append to it
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path, sep='\t')
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data

        # Deduplicate (guards against crash-and-retry leaving duplicates)
        dedup_cols = ['batch_id', 'sample_ID', 'gene', 'pos', 'alt']
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
        if len(combined) < before:
            logger.warning(f"Removed {before - len(combined)} duplicate rows during consolidation")

        # Save consolidated file
        combined.to_csv(output_path, sep='\t', index=False)
        logger.info(f"Saved {len(combined)} total mutations to {output_path}")

        return combined, newly_processed

    return pd.DataFrame(), []


def get_unique_samples(mutations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique sample information from mutations dataframe.

    Args:
        mutations_df: DataFrame with consolidated mutations

    Returns:
        DataFrame with unique batch_id/sample_ID combinations
    """
    if mutations_df.empty:
        return pd.DataFrame()

    return mutations_df[['batch_id', 'sample_ID']].drop_duplicates()


def get_mutation_summary(mutations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for mutations.

    Args:
        mutations_df: DataFrame with consolidated mutations

    Returns:
        DataFrame with mutation counts by gene and variant type
    """
    if mutations_df.empty:
        return pd.DataFrame()

    summary = mutations_df.groupby(['gene', 'variant_type']).agg({
        'sample_ID': 'nunique',
        'batch_id': 'nunique'
    }).reset_index()

    summary.columns = ['gene', 'variant_type', 'n_samples', 'n_batches']
    summary = summary.sort_values(['gene', 'n_samples'], ascending=[True, False])

    return summary


def filter_mutations(
    mutations_df: pd.DataFrame,
    min_vaf: float = 0.001,
    max_pvalue: float = 0.05,
    exclude_flagged: bool = True,
    vaf_column: str = 'allele.frequency'
) -> pd.DataFrame:
    """
    Filter mutations by VAF, P-value, and flags.

    Args:
        mutations_df: DataFrame with mutations
        min_vaf: Minimum VAF threshold (default 0.001 = 0.1% for emerging CHIP detection)
        max_pvalue: Maximum P-value (default 0.05)
        exclude_flagged: Exclude mutations with non-empty flags
        vaf_column: Column to use for VAF filtering
                    'allele.frequency' - matches R analysis (default)
                    'SSCS.allele.frequency' - more conservative

    Returns:
        Filtered DataFrame
    """
    if mutations_df.empty:
        return mutations_df

    df = mutations_df.copy()

    # Filter by VAF using specified column
    # Use > (not >=) to match R analysis behavior
    if vaf_column in df.columns:
        df = df[df[vaf_column] > min_vaf]
    elif 'SSCS.allele.frequency' in df.columns:
        # Fallback to SSCS if specified column not found
        df = df[df['SSCS.allele.frequency'] > min_vaf]

    # Filter by P-value
    if 'P-value' in df.columns:
        df = df[df['P-value'] <= max_pvalue]

    # Exclude flagged mutations
    if exclude_flagged and 'flags' in df.columns:
        df = df[(df['flags'].isna()) | (df['flags'] == '') | (df['flags'].astype(str) == 'nan')]

    return df


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test with sample batches
    test_batches = {
        'KG001_01.22.25': '/Volumes/Seq_SSD/smMIP/KG001_01.22.25',
        '10_23_25': '/Volumes/Seq_SSD/smMIP/10_23_25'
    }

    # Check which batches exist
    existing_batches = {k: v for k, v in test_batches.items() if os.path.isdir(v)}

    if existing_batches:
        print(f"Testing with batches: {list(existing_batches.keys())}")

        output = '/Volumes/Seq_SSD/smMIP/Master_Output/test_mutations.txt'
        df, processed = consolidate_mutations(existing_batches, output)

        print(f"\nProcessed batches: {processed}")
        print(f"Total mutations: {len(df)}")

        if not df.empty:
            print("\nUnique genes:")
            print(df['gene'].value_counts().head(10))

            print("\nSample summary:")
            print(get_unique_samples(df).head())

        # Cleanup test file
        if os.path.exists(output):
            os.remove(output)
    else:
        print("No test batches found")
        sys.exit(1)
