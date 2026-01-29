# smMIP Data Compilation Pipeline
This is a follow up to the smMIP-Pipeline outputs. Found at - https://github.com/wwadley-lucas/smMIP-Pipeline
A Python pipeline for consolidating and analyzing smMIP (single-molecule Molecular Inversion Probe) sequencing data for CHIP (Clonal Hematopoiesis of Indeterminate Potential) detection.

## Overview

This pipeline aggregates mutation data from multiple smMIP sequencing batches into a unified Master_Output, generates visualizations, and performs statistical analyses.

## Components

| Module | Description |
|--------|-------------|
| `master_output_manager.py` | Main orchestrator that coordinates all pipeline steps |
| `mutation_consolidator.py` | Merges `called_mutations.txt` files from multiple batches |
| `clinical_data_extractor.py` | Extracts and links clinical metadata from MetaData.txt files |
| `sample_index_builder.py` | Builds comprehensive sample index with batch tracking |
| `visualization_generator.py` | Generates oncoplots, PCA plots, VAF regression plots |

## Features

- **Batch consolidation**: Merge mutations from multiple sequencing batches
- **Clinical linkage**: Link mutations to patient metadata (age, sex, MPN status)
- **VAF filtering**: Configurable VAF threshold (default 0.1% for emerging CHIP detection)
- **Visualizations**:
  - Oncoplot (mutation landscape)
  - PCA plot (mutation VAF profiles, colored by MPN type)
  - VAF vs Age regression plots
  - Gene co-occurrence analysis

## Usage

```bash
# Run full pipeline
python master_output_manager.py

# Force rebuild (ignore cached data)
python master_output_manager.py --force
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn
- plotly
- kaleido (for PDF/PNG export)
- openpyxl (for Excel export)

## Output Structure

```
Master_Output/
├── master_called_mutations.txt    # Consolidated mutations
├── clinical_linkage.csv           # Clinical metadata
├── master_sample_index.xlsx       # Sample tracking
├── output/
│   ├── oncoplots/                 # Mutation landscape plots
│   ├── pca_plots/                 # PCA visualizations
│   └── regression_plots/          # VAF vs Age, gene pairs
└── stats/                         # Statistical analyses
```

## Configuration

Key parameters in `mutation_consolidator.py` and `visualization_generator.py`:
- `min_vaf`: Minimum VAF threshold (default: 0.001 = 0.1%)
- `max_pvalue`: P-value cutoff (default: 0.05)
- `top_n_genes`: Number of genes in oncoplot (default: 30)
