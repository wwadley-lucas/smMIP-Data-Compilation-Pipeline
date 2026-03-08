#!/usr/bin/env python3
"""
master_output_manager.py

Main entry point for Master_Output management.
Orchestrates batch discovery, mutation consolidation, clinical extraction,
sample indexing, and visualization generation.

Usage:
    python master_output_manager.py                    # Auto-discover and update
    python master_output_manager.py --force-rebuild    # Force full rebuild
    python master_output_manager.py --batch KG001_01.22.25  # Process specific batch
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from clinical_data_extractor import build_clinical_linkage, find_order_sheet, find_clinical_data
from mutation_consolidator import consolidate_mutations, find_mutations_file, filter_mutations
from sample_index_builder import build_sample_index, update_sample_index
from visualization_generator import generate_all_visualizations

# Configuration — set SMMIP_ROOT env var or pass --root on CLI
# No hardcoded user-specific paths; defaults to current working directory if env var is unset
SMMIP_ROOT = os.environ.get('SMMIP_ROOT', os.getcwd())
MASTER_OUTPUT_DIR = os.path.join(SMMIP_ROOT, 'Master_Output')

# External data overrides for specific batches
# Maps batch_id to dict with 'mutations' and/or 'metadata' paths
# Paths are relative to SMMIP_ROOT (resolved via --root CLI arg or SMMIP_ROOT env var)
EXTERNAL_DATA_OVERRIDES = {
    '10_23_25': {
        'mutations': os.path.join(SMMIP_ROOT, 'FINAL_OUT', 'called_mutations.txt'),
        'metadata': os.path.join(SMMIP_ROOT, 'FINAL_OUT', 'MetaData.txt')
    }
}

# Batch naming patterns
BATCH_PATTERNS = [
    r'^KG\d{3}_\d{2}\.\d{2}\.\d{2}$',       # KGxxx_MM.DD.YY
    r'^WW\d{3}_\d{2}\.\d{2}\.\d{2}$',       # WWxxx_MM.DD.YY
    r'^AA\d{3}_\d{2}\.\d{2}\.\d{2}$',       # AAxxx_MM.DD.YY
    r'^\d{1,2}_\d{1,2}_\d{2,4}$',           # M_D_YY or MM_DD_YYYY (date format)
]


def ensure_directory_structure() -> None:
    """Create Master_Output directory structure if it doesn't exist."""
    directories = [
        MASTER_OUTPUT_DIR,
        os.path.join(MASTER_OUTPUT_DIR, 'output', 'oncoplots'),
        os.path.join(MASTER_OUTPUT_DIR, 'output', 'regression_plots', 'vaf_vs_age'),
        os.path.join(MASTER_OUTPUT_DIR, 'output', 'regression_plots', 'gene_pairs'),
        os.path.join(MASTER_OUTPUT_DIR, 'output', 'pca_plots'),
        os.path.join(MASTER_OUTPUT_DIR, 'stats'),
        os.path.join(MASTER_OUTPUT_DIR, 'logs'),
    ]

    for d in directories:
        os.makedirs(d, exist_ok=True)
        logger.debug(f"Ensured directory exists: {d}")


def is_valid_batch_name(name: str) -> bool:
    """Check if directory name matches batch naming patterns."""
    for pattern in BATCH_PATTERNS:
        if re.match(pattern, name):
            return True
    return False


def has_mutation_data(batch_dir: str) -> bool:
    """Check if batch directory contains mutation data."""
    return find_mutations_file(batch_dir) is not None


def discover_batches() -> Dict[str, str]:
    """
    Discover all valid batch directories in smMIP root.

    Returns:
        Dict mapping batch_id to directory path
    """
    batches = {}

    if not os.path.isdir(SMMIP_ROOT):
        logger.error(f"smMIP root directory not found: {SMMIP_ROOT}")
        return batches

    for item in os.listdir(SMMIP_ROOT):
        item_path = os.path.join(SMMIP_ROOT, item)

        # Must be a directory
        if not os.path.isdir(item_path):
            continue

        # Skip Master_Output and Universal directories
        if item in ['Master_Output', 'Universal']:
            continue

        # Check if it matches batch naming pattern
        if is_valid_batch_name(item):
            # Verify it has mutation data
            if has_mutation_data(item_path):
                batches[item] = item_path
                logger.debug(f"Found batch: {item}")
            else:
                logger.debug(f"Batch {item} has no mutation data, skipping")

    logger.info(f"Discovered {len(batches)} batches with mutation data")
    return batches


def load_processed_batches() -> Set[str]:
    """Load set of already processed batch IDs from manifest."""
    manifest_path = os.path.join(MASTER_OUTPUT_DIR, 'processed_batches.json')

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
                return set(data.get('processed', []))
        except Exception as e:
            logger.warning(f"Error loading manifest: {e}")

    return set()


def save_processed_batches(processed: Set[str]) -> None:
    """Save processed batch IDs to manifest."""
    manifest_path = os.path.join(MASTER_OUTPUT_DIR, 'processed_batches.json')

    data = {
        'processed': list(processed),
        'last_updated': datetime.now().isoformat(),
        'total_batches': len(processed)
    }

    with open(manifest_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated manifest: {len(processed)} batches processed")


def discover_new_batches(all_batches: Dict[str, str]) -> Dict[str, str]:
    """
    Find batches that haven't been processed yet.

    Args:
        all_batches: Dict of all discovered batches

    Returns:
        Dict of new (unprocessed) batches
    """
    processed = load_processed_batches()
    new_batches = {k: v for k, v in all_batches.items() if k not in processed}

    if new_batches:
        logger.info(f"Found {len(new_batches)} new batches to process: {list(new_batches.keys())}")
    else:
        logger.info("No new batches to process")

    return new_batches


def _get_update_logger() -> logging.Logger:
    """Return a logger with RotatingFileHandler for the update log."""
    update_logger = logging.getLogger('master_output_update_log')
    if not update_logger.handlers:
        from logging.handlers import RotatingFileHandler
        log_path = os.path.join(MASTER_OUTPUT_DIR, 'logs', 'update_log.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=3  # 10 MB, 3 backups
        )
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        update_logger.addHandler(handler)
        update_logger.setLevel(logging.INFO)
    return update_logger


def log_update(message: str) -> None:
    """Append message to update log with automatic rotation (10 MB, 3 backups)."""
    _get_update_logger().info(message)


def update_master_output(
    force_rebuild: bool = False,
    specific_batches: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Run full Master_Output update pipeline.

    Args:
        force_rebuild: If True, reprocess all batches
        specific_batches: List of specific batch IDs to process (optional)

    Returns:
        Tuple of (success, message)
    """
    logger.info("=" * 60)
    logger.info("Starting Master_Output update")
    logger.info("=" * 60)

    # Ensure directory structure
    ensure_directory_structure()

    # Discover batches
    all_batches = discover_batches()

    if not all_batches:
        msg = "No batches found with mutation data"
        logger.warning(msg)
        return False, msg

    # Determine which batches to process
    if specific_batches:
        batches_to_process = {k: v for k, v in all_batches.items() if k in specific_batches}
        if not batches_to_process:
            msg = f"Specified batches not found: {specific_batches}"
            logger.error(msg)
            return False, msg
    elif force_rebuild:
        batches_to_process = all_batches
        # Clear manifest for rebuild
        save_processed_batches(set())
        logger.info("Force rebuild: processing all batches")
    else:
        batches_to_process = discover_new_batches(all_batches)

    if not batches_to_process:
        msg = "No new batches to process"
        logger.info(msg)
        log_update(msg)
        return True, msg

    log_update(f"Processing {len(batches_to_process)} batches: {list(batches_to_process.keys())}")

    # =========================================================================
    # Step 1: Consolidate mutations
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Step 1: Consolidating mutations")
    logger.info("-" * 40)

    mutations_path = os.path.join(MASTER_OUTPUT_DIR, 'master_called_mutations.txt')
    processed_batches = load_processed_batches() if not force_rebuild else set()

    # Build external mutations dict from overrides
    external_mutations = {}
    for batch_id, override in EXTERNAL_DATA_OVERRIDES.items():
        if 'mutations' in override:
            external_mutations[batch_id] = override['mutations']

    mutations_df, newly_processed = consolidate_mutations(
        all_batches if force_rebuild else batches_to_process,
        mutations_path,
        processed_batches,
        external_mutations=external_mutations
    )

    if mutations_df.empty:
        msg = "No mutations consolidated"
        logger.warning(msg)
        return False, msg

    logger.info(f"Total mutations: {len(mutations_df)}")

    # =========================================================================
    # Step 2: Extract clinical data
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Step 2: Extracting clinical data")
    logger.info("-" * 40)

    clinical_path = os.path.join(MASTER_OUTPUT_DIR, 'clinical_linkage.csv')

    # Build external metadata dict from overrides
    external_metadata = {}
    for batch_id, override in EXTERNAL_DATA_OVERRIDES.items():
        if 'metadata' in override:
            external_metadata[batch_id] = override['metadata']

    clinical_df = build_clinical_linkage(all_batches, clinical_path, external_metadata=external_metadata)

    if not clinical_df.empty:
        logger.info(f"Clinical data extracted: {len(clinical_df)} records")
    else:
        logger.warning("No clinical data extracted (order sheets may be missing)")
        clinical_df = pd.DataFrame()

    # =========================================================================
    # Step 3: Build sample index
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Step 3: Building sample index")
    logger.info("-" * 40)

    index_path = os.path.join(MASTER_OUTPUT_DIR, 'master_sample_index.xlsx')
    sample_index = build_sample_index(all_batches, mutations_df, index_path)

    logger.info(f"Sample index: {len(sample_index)} samples")

    # =========================================================================
    # Step 4: Generate visualizations
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Step 4: Generating visualizations")
    logger.info("-" * 40)

    output_dir = os.path.join(MASTER_OUTPUT_DIR, 'output')

    try:
        # Filter mutations for visualization
        filtered_mutations = filter_mutations(mutations_df)
        logger.info(f"Filtered mutations for visualization: {len(filtered_mutations)}")

        generate_all_visualizations(filtered_mutations, clinical_df, output_dir)
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Step 5: Update manifest
    # =========================================================================
    logger.info("-" * 40)
    logger.info("Step 5: Updating manifest")
    logger.info("-" * 40)

    processed_batches.update(newly_processed)
    save_processed_batches(processed_batches)

    # Log completion
    msg = f"Update complete: processed {len(newly_processed)} batches, {len(mutations_df)} total mutations"
    logger.info(msg)
    log_update(msg)

    logger.info("=" * 60)
    logger.info("Master_Output update finished successfully")
    logger.info("=" * 60)

    return True, msg


def print_status() -> None:
    """Print current status of Master_Output."""
    print("\n" + "=" * 60)
    print("Master_Output Status")
    print("=" * 60)

    # Check directory exists
    if not os.path.isdir(MASTER_OUTPUT_DIR):
        print("Master_Output directory does not exist")
        return

    # Check manifest
    processed = load_processed_batches()
    print(f"\nProcessed batches: {len(processed)}")
    for batch in sorted(processed):
        print(f"  - {batch}")

    # Check files
    files = {
        'master_called_mutations.txt': 'Mutations',
        'master_sample_index.xlsx': 'Sample Index',
        'clinical_linkage.csv': 'Clinical Linkage',
        'processed_batches.json': 'Manifest'
    }

    print("\nFiles:")
    for filename, label in files.items():
        path = os.path.join(MASTER_OUTPUT_DIR, filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {label}: {size:,} bytes")
        else:
            print(f"  ✗ {label}: not found")

    # Check visualizations
    viz_dir = os.path.join(MASTER_OUTPUT_DIR, 'output')
    if os.path.isdir(viz_dir):
        viz_count = sum(1 for _ in Path(viz_dir).rglob('*') if _.is_file())
        print(f"\nVisualization files: {viz_count}")

    # Discover available batches
    all_batches = discover_batches()
    new_batches = discover_new_batches(all_batches)
    print(f"\nAvailable batches: {len(all_batches)}")
    print(f"New (unprocessed): {len(new_batches)}")

    print("=" * 60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Master_Output Manager for smMIP Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_output_manager.py              # Auto-update with new batches
  python master_output_manager.py --status     # Show current status
  python master_output_manager.py --force      # Force full rebuild
  python master_output_manager.py --batch KG001_01.22.25  # Process specific batch
        """
    )

    parser.add_argument(
        '--root',
        default=None,
        help='Root directory for smMIP data (overrides SMMIP_ROOT env var)'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show current status without updating'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force rebuild of all data (reprocess all batches)'
    )

    parser.add_argument(
        '--batch', '-b',
        nargs='+',
        help='Process specific batch(es) by ID'
    )

    parser.add_argument(
        '--discover', '-d',
        action='store_true',
        help='Only discover batches, do not process'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    parser.add_argument(
        '--metadata', '-m',
        nargs=2,
        metavar=('BATCH_ID', 'PATH'),
        action='append',
        help='Override metadata file for a batch (can be used multiple times)'
    )

    parser.add_argument(
        '--mutations',
        nargs=2,
        metavar=('BATCH_ID', 'PATH'),
        action='append',
        help='Override mutations file for a batch (can be used multiple times)'
    )

    args = parser.parse_args()

    # Apply --root override
    global SMMIP_ROOT, MASTER_OUTPUT_DIR, EXTERNAL_DATA_OVERRIDES
    if args.root:
        SMMIP_ROOT = args.root
        MASTER_OUTPUT_DIR = os.path.join(SMMIP_ROOT, 'Master_Output')
        # Recompute overrides that depend on SMMIP_ROOT
        EXTERNAL_DATA_OVERRIDES = {
            '10_23_25': {
                'mutations': os.path.join(SMMIP_ROOT, 'FINAL_OUT', 'called_mutations.txt'),
                'metadata': os.path.join(SMMIP_ROOT, 'FINAL_OUT', 'MetaData.txt')
            }
        }

    # SEC-001: Validate and process CLI overrides into EXTERNAL_DATA_OVERRIDES
    if args.metadata:
        for batch_id, path in args.metadata:
            if not os.path.exists(path):
                logger.error(f"--metadata path does not exist: {path}")
                exit(1)
            if batch_id not in EXTERNAL_DATA_OVERRIDES:
                EXTERNAL_DATA_OVERRIDES[batch_id] = {}
            EXTERNAL_DATA_OVERRIDES[batch_id]['metadata'] = path

    if args.mutations:
        for batch_id, path in args.mutations:
            if not os.path.exists(path):
                logger.error(f"--mutations path does not exist: {path}")
                exit(1)
            if batch_id not in EXTERNAL_DATA_OVERRIDES:
                EXTERNAL_DATA_OVERRIDES[batch_id] = {}
            EXTERNAL_DATA_OVERRIDES[batch_id]['mutations'] = path

    # Set log level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Handle commands
    if args.status:
        print_status()
        return

    if args.discover:
        batches = discover_batches()
        print(f"\nDiscovered {len(batches)} batches:")
        for batch_id, path in sorted(batches.items()):
            processed = batch_id in load_processed_batches()
            status = "✓ processed" if processed else "○ new"
            print(f"  {batch_id}: {status}")
        return

    # Run update
    success, message = update_master_output(
        force_rebuild=args.force,
        specific_batches=args.batch
    )

    if not success:
        logger.error(f"Update failed: {message}")
        exit(1)


if __name__ == '__main__':
    main()
