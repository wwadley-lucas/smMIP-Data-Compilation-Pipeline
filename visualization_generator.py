#!/usr/bin/env python3
"""
visualization_generator.py

Generate all visualizations for Master_Output:
- Oncoplots (master overview of mutations)
- VAF vs Age regression plots (one per gene)
- Gene pair VAF correlation plots (significant co-occurring pairs)
- PCA plots (clinical + mutation features)

All plots exported in PDF, PNG, and interactive HTML formats.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import logging
import warnings

from clinical_data_extractor import normalize_sample_id

# CLR-001: Filter only specific warning categories instead of blanket suppression
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

logger = logging.getLogger(__name__)

# Variant type color scheme (from CHIP_analysis.R)
# Keys are canonical display categories produced by normalize_variant_type().
# Raw VEP annotations (e.g., missense_variant, stop_gained) are mapped to
# these canonical categories before color lookup, so only one entry per
# category is needed.
VARIANT_COLORS = {
    'Missense': '#3498db',      # Blue  (missense_variant)
    'Nonsense': '#e74c3c',      # Red   (stop_gained)
    'Deletion': '#8e44ad',      # Dark Purple  (frameshift_variant)
    'Insertion': '#9b59b6',     # Purple  (inframe_insertion)
    'Indel': '#a569bd',         # Light Purple  (inframe_deletion)
    'Splice_Site': '#f39c12',   # Orange  (splice_region/donor/acceptor_variant)
    'Synonymous': '#95a5a6',    # Gray  (synonymous_variant)
    'Intronic': '#7f8c8d',      # Dark Gray  (intron_variant)
    'Non_Coding': '#bdc3c7',    # Light Gray  (5'/3'_prime_UTR_variant)
    'Other': '#2c3e50',         # Very Dark Blue
}

# VAF color scale (red gradient)
VAF_COLORS = ['#ffffff', '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d', '#a50f15']
VAF_BREAKS = [0, 0.05, 0.1, 0.2, 0.5, 1.0]

# Gene risk categories
GENE_RISK = {
    'very_high': ['TP53'],
    'high': ['ASXL1', 'IDH1', 'IDH2', 'EZH2', 'SF3B1', 'SRSF2', 'U2AF1', 'JAK2',
             'MPL', 'CALR', 'NRAS', 'KRAS', 'KIT', 'FLT3', 'RUNX1', 'CEBPA',
             'WT1', 'GATA1', 'GATA2', 'NPM1', 'PPM1D', 'PHF6', 'BCOR', 'SETBP1'],
    'intermediate_high': ['CBL', 'ZRSR2', 'STAG2', 'PTPN11'],
    'intermediate': ['DNMT3A', 'TET2', 'GNAS', 'SMC1A', 'SMC3', 'RAD21'],
    'low': ['UTP23']
}


def normalize_variant_type(variant_type: str) -> str:
    """
    Normalize variant type strings into canonical categories for consistent coloring.

    Maps the diverse variant type annotations found in called_mutations.txt
    (e.g., 'missense_variant', 'stop_gained', 'frameshift_variant') to a small
    set of display categories (e.g., 'Missense', 'Nonsense', 'Deletion') that
    correspond to entries in VARIANT_COLORS.

    Args:
        variant_type: Raw variant type string from mutation data.  May be a
            VEP-style annotation (e.g., 'missense_variant') or a shorthand
            label (e.g., 'Missense').

    Returns:
        Canonical variant category string.  One of: 'Missense', 'Nonsense',
        'Deletion', 'Insertion', 'Indel', 'Splice_Site', 'Synonymous',
        'Intronic', 'Non_Coding', or 'Other'.  Returns 'Other' for NaN
        or unrecognized values.
    """
    if pd.isna(variant_type):
        return 'Other'

    vt = str(variant_type).lower()

    if 'missense' in vt:
        return 'Missense'
    elif 'stop_gained' in vt or 'nonsense' in vt:
        return 'Nonsense'
    elif 'frameshift' in vt or ('deletion' in vt and 'inframe' not in vt):
        return 'Deletion'
    elif 'inframe_insertion' in vt or 'insertion' in vt:
        return 'Insertion'
    elif 'inframe_deletion' in vt or 'indel' in vt:
        return 'Indel'
    elif 'splice' in vt:
        return 'Splice_Site'
    elif 'synonymous' in vt:
        return 'Synonymous'
    elif 'intron' in vt:
        return 'Intronic'
    elif 'utr' in vt or 'non_coding' in vt:
        return 'Non_Coding'
    else:
        return 'Other'


def _prepare_oncoplot_data(
    mutations_df: pd.DataFrame,
    top_n_genes: int,
    vaf_column: str
) -> Optional[Tuple[pd.DataFrame, List[str], List[str], List[List[Optional[str]]],
                     List[List[str]], pd.Series]]:
    """
    Filter mutations and build the gene-sample matrix for an oncoplot.

    Normalizes sample IDs, selects the top N genes by sample count, constructs
    a variant-type matrix (genes x samples) picking the highest-severity variant
    per cell, and builds matching hover text.

    Args:
        mutations_df: DataFrame with consolidated mutations (already pre-filtered
            by the manager).
        top_n_genes: Number of top mutated genes to include.
        vaf_column: Column name used for VAF display in hover text.

    Returns:
        A tuple of (df, genes, samples, matrix, hover_text, gene_sample_counts)
        where *matrix* and *hover_text* are lists-of-lists aligned to
        genes (rows) x samples (cols).  Returns ``None`` if the input is empty.
    """
    df = mutations_df.copy()

    if df.empty:
        logger.warning("No mutations pass filters for oncoplot")
        return None

    # Normalize sample IDs
    df['sample_normalized'] = df['sample_ID'].apply(normalize_sample_id)

    # Get top genes by sample count (sorted descending - most mutated first)
    gene_sample_counts = df.groupby('gene')['sample_normalized'].nunique().sort_values(ascending=False)
    top_genes = gene_sample_counts.head(top_n_genes).index.tolist()

    # Filter to top genes
    df = df[df['gene'].isin(top_genes)]

    # Get unique samples, sorted by mutation count (descending)
    sample_mutation_counts = df.groupby('sample_normalized')['gene'].nunique().sort_values(ascending=False)
    samples = sample_mutation_counts.index.tolist()

    # Genes are already sorted by frequency (most at top)
    genes = top_genes

    # Create mutation matrix
    severity_order = ['Nonsense', 'Deletion', 'Insertion', 'Indel', 'Splice_Site',
                      'Missense', 'Synonymous', 'Intronic', 'Non_Coding', 'Other']

    df['variant_normalized'] = df['variant_type'].apply(normalize_variant_type)

    matrix = []
    hover_text = []

    for gene in genes:
        gene_row = []
        hover_row = []
        for sample in samples:
            subset = df[(df['gene'] == gene) & (df['sample_normalized'] == sample)]
            if len(subset) > 0:
                # Get highest severity variant
                subset = subset.copy()
                subset['severity'] = subset['variant_normalized'].apply(
                    lambda x: severity_order.index(x) if x in severity_order else len(severity_order)
                )
                best = subset.loc[subset['severity'].idxmin()]
                gene_row.append(best['variant_normalized'])

                vaf = best.get(vaf_column, best.get('SSCS.allele.frequency', 0))
                vaf = float(vaf) if pd.notna(vaf) else 0

                protein = best.get('protein', '')
                if pd.notna(protein) and ':' in str(protein):
                    protein = str(protein).split(':')[-1].split(';')[0]
                hover_row.append(f"{gene}<br>{best['variant_normalized']}<br>VAF: {vaf:.3f}<br>{protein}")
            else:
                gene_row.append(None)
                hover_row.append("")
        matrix.append(gene_row)
        hover_text.append(hover_row)

    return df, genes, samples, matrix, hover_text, gene_sample_counts


def _build_clinical_annotations(
    clinical_df: Optional[pd.DataFrame],
    samples: List[str]
) -> Optional[Dict]:
    """
    Build clinical annotation arrays and color mappings for an oncoplot.

    Currently the oncoplot does not include a clinical annotation bar, so this
    function is a no-op placeholder.  It returns ``None``, which the orchestrator
    interprets as "skip clinical annotations".  When clinical annotation support
    is added, this function should return a dict with keys like
    ``{'values': [...], 'colors': {...}}``.

    Args:
        clinical_df: DataFrame with clinical data, or ``None``.
        samples: Ordered list of sample IDs matching the heatmap columns.

    Returns:
        ``None`` (no annotations available).
    """
    # Placeholder — clinical annotation bar not yet implemented in oncoplot.
    return None


def _render_oncoplot_heatmap(
    fig,
    matrix: List[List[Optional[str]]],
    hover_text: List[List[str]],
    genes: List[str],
    samples: List[str]
) -> None:
    """
    Add the main gene-by-sample heatmap trace to the oncoplot figure.

    Converts the variant-type matrix into a numeric z-array with a discrete
    colorscale derived from :data:`VARIANT_COLORS`, then adds a
    ``plotly.graph_objects.Heatmap`` trace at row 2, col 1.

    Args:
        fig: Plotly ``Figure`` (2x2 subplot grid) to add the trace to.
        matrix: Gene x sample matrix of variant-type strings (or ``None``).
        hover_text: Matching hover-text matrix.
        genes: Ordered gene list (y-axis).
        samples: Ordered sample list (x-axis).
    """
    import plotly.graph_objects as go

    severity_order = ['Nonsense', 'Deletion', 'Insertion', 'Indel', 'Splice_Site',
                      'Missense', 'Synonymous', 'Intronic', 'Non_Coding', 'Other']

    # Convert variants to numeric for coloring
    variant_to_num = {v: i+1 for i, v in enumerate(severity_order)}

    z_data = []
    for row in matrix:
        z_row = [variant_to_num.get(v, 0) if v else 0 for v in row]
        z_data.append(z_row)

    # Custom colorscale (discrete colors)
    n_variants = len(severity_order)
    colorscale = [[0, '#f5f5f5']]  # No mutation (light gray)
    for i, vtype in enumerate(severity_order):
        color = VARIANT_COLORS.get(vtype, '#2c3e50')
        # Create discrete color bands
        start = (i + 0.5) / (n_variants + 1)
        end = (i + 1.5) / (n_variants + 1)
        colorscale.append([start, color])
        colorscale.append([end, color])
    colorscale.append([1.0, VARIANT_COLORS.get('Other', '#2c3e50')])

    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=list(range(len(samples))),
            y=genes,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            colorscale=colorscale,
            showscale=False,
            xgap=1,
            ygap=1,
            zmin=0,
            zmax=n_variants + 1
        ),
        row=2, col=1
    )


def _render_frequency_bars(
    fig,
    df: pd.DataFrame,
    genes: List[str],
    samples: List[str],
    gene_sample_counts: pd.Series
) -> None:
    """
    Add top and side frequency bar charts to the oncoplot figure.

    - **Top bar** (row 1, col 1): total mutations per sample.
    - **Side bar** (row 2, col 2): number of unique samples mutated per gene.

    Args:
        fig: Plotly ``Figure`` (2x2 subplot grid) to add traces to.
        df: Filtered mutations DataFrame (must contain ``sample_normalized``
            and ``gene`` columns).
        genes: Ordered gene list matching the heatmap y-axis.
        samples: Ordered sample list matching the heatmap x-axis.
        gene_sample_counts: Series mapping gene -> unique sample count.
    """
    import plotly.graph_objects as go

    # Top bar chart - mutations per sample (row 1, col 1)
    sample_counts = df.groupby('sample_normalized')['gene'].count()
    sample_counts = sample_counts.reindex(samples).fillna(0)

    fig.add_trace(
        go.Bar(
            x=list(range(len(samples))),
            y=sample_counts.values,
            marker_color='#4472C4',
            showlegend=False,
            hovertemplate='%{y} mutations<extra></extra>'
        ),
        row=1, col=1
    )

    # Right bar chart - samples per gene (row 2, col 2)
    gene_counts = [gene_sample_counts[gene] for gene in genes]

    fig.add_trace(
        go.Bar(
            y=list(range(len(genes))),
            x=gene_counts,
            orientation='h',
            marker_color='#27ae60',
            showlegend=False,
            hovertemplate='%{x} samples<extra></extra>'
        ),
        row=2, col=2
    )


def _assemble_oncoplot_figure(
    fig,
    genes: List[str],
    samples: List[str],
    output_dir: str,
    filename_base: str
) -> None:
    """
    Apply final layout, add legend and title, and save the oncoplot figure.

    Configures axis labels, tick formatting, dynamic figure sizing, a variant-type
    legend at the bottom, and a bold title.  Saves the figure as interactive HTML,
    plus static PDF and PNG (if kaleido is installed).

    Args:
        fig: Fully populated Plotly ``Figure`` with all traces added.
        genes: Ordered gene list (used for axis ticks and sizing).
        samples: Ordered sample list (used for axis ticks and sizing).
        output_dir: Directory to save output files.
        filename_base: Base filename (without extension) for saved files.
    """
    # Layout configuration
    MIN_FIG_HEIGHT = 600
    MIN_FIG_WIDTH = 1000
    PX_PER_GENE = 35
    PX_PER_SAMPLE = 18
    FIG_PADDING = 250

    fig_height = max(MIN_FIG_HEIGHT, PX_PER_GENE * len(genes) + FIG_PADDING)
    fig_width = max(MIN_FIG_WIDTH, PX_PER_SAMPLE * len(samples) + FIG_PADDING)

    fig.update_layout(
        height=fig_height,
        width=fig_width,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=80, r=120, t=80, b=150),
        showlegend=False
    )

    # Top bar chart axes
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text='# Mutations', row=1, col=1)

    # Main heatmap axes
    fig.update_xaxes(
        ticktext=samples,
        tickvals=list(range(len(samples))),
        tickangle=45,
        tickfont=dict(size=9),
        row=2, col=1
    )
    fig.update_yaxes(
        ticktext=genes,
        tickvals=list(range(len(genes))),
        tickfont=dict(size=10),
        row=2, col=1
    )

    # Right bar chart axes
    fig.update_xaxes(title_text='# Samples', row=2, col=2)
    fig.update_yaxes(
        showticklabels=False,
        row=2, col=2
    )

    # Add legend as custom shapes/annotations at the bottom
    legend_items = ['Missense', 'Nonsense', 'Deletion', 'Splice_Site', 'Insertion', 'Other']
    legend_y = -0.12
    legend_start_x = 0.02
    legend_spacing = 0.16

    for i, vtype in enumerate(legend_items):
        color = VARIANT_COLORS.get(vtype, '#2c3e50')
        x_pos = legend_start_x + i * legend_spacing

        # Add colored square
        fig.add_annotation(
            x=x_pos,
            y=legend_y,
            xref='paper',
            yref='paper',
            text=f'<span style="color:{color}; font-size:16px">■</span> {vtype}',
            showarrow=False,
            font=dict(size=10),
            xanchor='left'
        )

    # Add title
    fig.add_annotation(
        x=0.5,
        y=1.02,
        xref='paper',
        yref='paper',
        text='<b>Mutation Landscape</b>',
        showarrow=False,
        font=dict(size=16),
        xanchor='center'
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # HTML (interactive)
    html_path = os.path.join(output_dir, f'{filename_base}.html')
    fig.write_html(html_path)
    logger.info(f"Saved oncoplot HTML: {html_path}")

    # PDF and PNG
    try:
        pdf_path = os.path.join(output_dir, f'{filename_base}.pdf')
        png_path = os.path.join(output_dir, f'{filename_base}.png')
        fig.write_image(pdf_path, format='pdf')
        fig.write_image(png_path, format='png', scale=2)
        logger.info(f"Saved oncoplot PDF: {pdf_path}")
        logger.info(f"Saved oncoplot PNG: {png_path}")
    except Exception as e:
        logger.warning(f"Could not save static images (install kaleido): {e}")


def create_oncoplot(
    mutations_df: pd.DataFrame,
    output_dir: str,
    filename_base: str = 'master_oncoplot',
    min_vaf: float = 0.001,
    max_pvalue: float = 0.05,
    top_n_genes: int = 30,
    vaf_column: str = 'allele.frequency'
) -> None:
    """
    Generate oncoplot showing mutation landscape.

    Layout matches R ComplexHeatmap style:
    - Top bar: Mutations per sample
    - Main: Gene x Sample heatmap (sorted by mutation frequency, most at top)
    - Right bar: Number of samples mutated per gene

    This function is a thin orchestrator that delegates to:
    - :func:`_prepare_oncoplot_data` — filtering and gene-sample matrix
    - :func:`_build_clinical_annotations` — clinical annotation bar (placeholder)
    - :func:`_render_oncoplot_heatmap` — main heatmap trace
    - :func:`_render_frequency_bars` — top and side bar charts
    - :func:`_assemble_oncoplot_figure` — layout, legend, and file export

    Args:
        mutations_df: DataFrame with consolidated mutations
        output_dir: Directory to save plots
        filename_base: Base filename for outputs
        min_vaf: Minimum VAF filter
        max_pvalue: Maximum P-value filter
        top_n_genes: Number of top mutated genes to show
        vaf_column: Column to use for VAF filtering
    """
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly not installed. Run: pip install plotly kaleido")
        return

    # NOTE: Filtering is handled by the manager (master_output_manager.py line 318)
    # via mutation_consolidator.filter_mutations() before data reaches this function.
    # Do NOT duplicate filtering here — the manager's pre-filter is authoritative.

    # 1. Data preparation
    result = _prepare_oncoplot_data(mutations_df, top_n_genes, vaf_column)
    if result is None:
        return
    df, genes, samples, matrix, hover_text, gene_sample_counts = result

    # 2. Clinical annotations (placeholder — not yet implemented)
    _build_clinical_annotations(None, samples)

    # 3. Build subplot grid
    # Oncoplot layout constants
    TOP_BAR_HEIGHT = 0.12        # Fraction of figure height for top bar chart
    MAIN_HEATMAP_HEIGHT = 0.88   # Fraction of figure height for main heatmap
    HEATMAP_WIDTH = 0.85         # Fraction of figure width for heatmap
    SIDE_BAR_WIDTH = 0.15        # Fraction of figure width for right bar chart
    VERTICAL_GAP = 0.05          # Gap between top bar and heatmap
    HORIZONTAL_GAP = 0.02        # Gap between heatmap and right bar

    # Create 2x2 subplot grid:
    # [top bar chart] [empty corner]
    # [main heatmap]  [right bar chart]
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[TOP_BAR_HEIGHT, MAIN_HEATMAP_HEIGHT],
        column_widths=[HEATMAP_WIDTH, SIDE_BAR_WIDTH],
        vertical_spacing=VERTICAL_GAP,
        horizontal_spacing=HORIZONTAL_GAP,
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],  # Top row
            [{"type": "heatmap"}, {"type": "bar"}]   # Bottom row
        ]
    )

    # 4. Render heatmap
    _render_oncoplot_heatmap(fig, matrix, hover_text, genes, samples)

    # 5. Render frequency bars
    _render_frequency_bars(fig, df, genes, samples, gene_sample_counts)

    # 6. Assemble final figure, add legend, and save
    _assemble_oncoplot_figure(fig, genes, samples, output_dir, filename_base)


def create_vaf_vs_age_plots(
    mutations_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Generate VAF vs Age regression plots for each gene with mutations.

    Args:
        mutations_df: DataFrame with consolidated mutations
        clinical_df: DataFrame with clinical data (must have D#, Age columns)
        output_dir: Directory to save plots
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        logger.error("Plotly not installed")
        return

    if clinical_df.empty or 'Age' not in clinical_df.columns:
        logger.warning("No clinical data with Age for VAF vs Age plots")
        return

    # Prepare data
    df = mutations_df.copy()
    df['sample_normalized'] = df['sample_ID'].apply(normalize_sample_id)

    # Create clinical lookup
    clinical_dict = {}
    for _, row in clinical_df.iterrows():
        d_num = str(row.get('D#', '')).strip()
        age = row.get('Age')
        if d_num and pd.notna(age):
            clinical_dict[d_num] = float(age)

    # Add age to mutations
    df['Age'] = df['sample_normalized'].map(clinical_dict)
    df = df[df['Age'].notna()]

    if df.empty:
        logger.warning("No mutations with linked age data")
        return

    # Get VAF column
    vaf_col = 'SSCS.allele.frequency' if 'SSCS.allele.frequency' in df.columns else 'allele.frequency'
    df['VAF'] = pd.to_numeric(df[vaf_col], errors='coerce')
    df = df[df['VAF'].notna()]

    # Get unique genes
    genes = df['gene'].unique()

    os.makedirs(output_dir, exist_ok=True)

    # MTH-008: First pass — collect regression p-values for FDR correction
    gene_stats = {}
    for gene in genes:
        gene_df = df[df['gene'] == gene]
        if len(gene_df) < 3 or gene_df['Age'].nunique() < 2:
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            gene_df['Age'], gene_df['VAF']
        )
        gene_stats[gene] = {
            'slope': slope, 'intercept': intercept,
            'r_value': r_value, 'p_value': p_value, 'std_err': std_err,
            'n': len(gene_df),
        }

    # Apply BH FDR correction across all tested genes
    if gene_stats:
        tested_genes = list(gene_stats.keys())
        raw_pvalues = [gene_stats[g]['p_value'] for g in tested_genes]
        _, fdr_pvalues, _, _ = multipletests(raw_pvalues, method='fdr_bh')
        for gene, q_val in zip(tested_genes, fdr_pvalues):
            gene_stats[gene]['q_value'] = q_val

    # Second pass — generate plots with FDR-corrected q-values
    for gene, gs in gene_stats.items():
        gene_df = df[df['gene'] == gene]

        # Create scatter plot with regression
        fig = px.scatter(
            gene_df,
            x='Age',
            y='VAF',
            trendline='ols',
            title=f'{gene} - VAF vs Age',
            labels={'Age': 'Age (years)', 'VAF': 'Variant Allele Frequency'},
            hover_data=['sample_normalized', 'variant_type']
        )

        fig.update_traces(marker=dict(size=10, color='#3498db'))

        # Add stats annotation (includes FDR q-value)
        fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=(f'R² = {gs["r_value"]**2:.3f}<br>p = {gs["p_value"]:.3e}'
                  f'<br>q(FDR) = {gs["q_value"]:.3e}<br>n = {gs["n"]}'),
            showarrow=False,
            align='left',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1
        )

        fig.update_layout(
            width=700,
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Save outputs
        base = f'{gene}_vaf_vs_age'
        fig.write_html(os.path.join(output_dir, f'{base}.html'))

        try:
            fig.write_image(os.path.join(output_dir, f'{base}.pdf'), format='pdf')
            fig.write_image(os.path.join(output_dir, f'{base}.png'), format='png', scale=2)
        except Exception as e:
            logger.warning(f"Could not save static images for {gene}: {e}")

    logger.info(f"Created VAF vs Age plots for {len(gene_stats)} genes")


def compute_fisher_cooccurrence(
    mutations_df: pd.DataFrame,
    output_dir: str,
    p_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Compute pairwise Fisher's exact test for gene co-occurrence.

    Args:
        mutations_df: DataFrame with consolidated mutations
        output_dir: Directory to save results
        p_threshold: P-value threshold for significance

    Returns:
        DataFrame with significant gene pairs
    """
    df = mutations_df.copy()
    df['sample_normalized'] = df['sample_ID'].apply(normalize_sample_id)

    # Build binary gene x sample matrix
    samples = df['sample_normalized'].unique()
    genes = df['gene'].unique()

    if len(genes) < 2:
        logger.warning("Need at least 2 genes for co-occurrence analysis")
        return pd.DataFrame()

    # BUG-009: Use crosstab instead of iterrows for O(n) vectorized construction
    gene_sample_matrix = pd.crosstab(
        df['sample_normalized'], df['gene']
    ).reindex(index=samples, columns=genes, fill_value=0).clip(upper=1)

    # Pairwise Fisher's exact test
    results = []
    n_samples = len(samples)

    for i, gene1 in enumerate(genes):
        for j, gene2 in enumerate(genes):
            if i >= j:
                continue

            # Build contingency table
            both = ((gene_sample_matrix[gene1] == 1) & (gene_sample_matrix[gene2] == 1)).sum()
            gene1_only = ((gene_sample_matrix[gene1] == 1) & (gene_sample_matrix[gene2] == 0)).sum()
            gene2_only = ((gene_sample_matrix[gene1] == 0) & (gene_sample_matrix[gene2] == 1)).sum()
            neither = ((gene_sample_matrix[gene1] == 0) & (gene_sample_matrix[gene2] == 0)).sum()

            contingency = [[both, gene1_only], [gene2_only, neither]]

            try:
                odds_ratio, p_value = fisher_exact(contingency)
            except Exception:
                continue

            results.append({
                'gene1': gene1,
                'gene2': gene2,
                'both': both,
                'gene1_only': gene1_only,
                'gene2_only': gene2_only,
                'neither': neither,
                'odds_ratio': odds_ratio,
                'p_value': p_value
            })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return pd.DataFrame()

    # Bonferroni correction
    n_tests = len(results_df)
    results_df['p_adjusted'] = results_df['p_value'] * n_tests
    results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)

    # Save full results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'fisher_cooccurrence.csv'), index=False)

    # Filter significant pairs using Bonferroni-corrected p-values
    # (raw p_value is retained for reference but significance must use p_adjusted)
    significant = results_df[results_df['p_adjusted'] < p_threshold].copy()
    significant = significant.sort_values('p_adjusted')
    significant.to_csv(os.path.join(output_dir, 'significant_pairs.csv'), index=False)

    logger.info(f"Found {len(significant)} significant gene pairs (p < {p_threshold})")

    return significant


def create_gene_pair_plots(
    mutations_df: pd.DataFrame,
    significant_pairs: pd.DataFrame,
    output_dir: str,
    max_pairs: int = 20
) -> None:
    """
    Generate VAF correlation plots for significant gene pairs.

    Args:
        mutations_df: DataFrame with consolidated mutations
        significant_pairs: DataFrame from compute_fisher_cooccurrence
        output_dir: Directory to save plots
        max_pairs: Maximum number of pairs to plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Plotly not installed")
        return

    if significant_pairs.empty:
        logger.info("No significant gene pairs to plot")
        return

    df = mutations_df.copy()
    df['sample_normalized'] = df['sample_ID'].apply(normalize_sample_id)

    vaf_col = 'SSCS.allele.frequency' if 'SSCS.allele.frequency' in df.columns else 'allele.frequency'

    os.makedirs(output_dir, exist_ok=True)

    pairs_to_plot = significant_pairs.head(max_pairs)

    for _, pair_row in pairs_to_plot.iterrows():
        gene1, gene2 = pair_row['gene1'], pair_row['gene2']

        # Get samples with both mutations
        gene1_data = df[df['gene'] == gene1][['sample_normalized', vaf_col]].copy()
        gene1_data.columns = ['sample', 'VAF1']
        gene1_data = gene1_data.groupby('sample')['VAF1'].max().reset_index()

        gene2_data = df[df['gene'] == gene2][['sample_normalized', vaf_col]].copy()
        gene2_data.columns = ['sample', 'VAF2']
        gene2_data = gene2_data.groupby('sample')['VAF2'].max().reset_index()

        merged = gene1_data.merge(gene2_data, on='sample', how='inner')

        if len(merged) < 3:
            continue

        # Create scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=merged['VAF1'],
            y=merged['VAF2'],
            mode='markers',
            marker=dict(size=10, color='#e74c3c'),
            text=merged['sample'],
            hovertemplate='Sample: %{text}<br>' + gene1 + ' VAF: %{x:.3f}<br>' + gene2 + ' VAF: %{y:.3f}'
        ))

        # Calculate correlation (Spearman: VAF is bounded [0,1], not necessarily normal)
        r, p = stats.spearmanr(merged['VAF1'], merged['VAF2'])

        fig.update_layout(
            title=f'{gene1} vs {gene2} VAF Correlation',
            xaxis_title=f'{gene1} VAF',
            yaxis_title=f'{gene2} VAF',
            width=600,
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=f'rho = {r:.3f}<br>p = {p:.3e}<br>n = {len(merged)}<br>Fisher p = {pair_row["p_value"]:.3e}',
            showarrow=False,
            align='left',
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1
        )

        # Save outputs
        base = f'{gene1}_vs_{gene2}'
        fig.write_html(os.path.join(output_dir, f'{base}.html'))

        try:
            fig.write_image(os.path.join(output_dir, f'{base}.pdf'), format='pdf')
            fig.write_image(os.path.join(output_dir, f'{base}.png'), format='png', scale=2)
        except Exception as e:
            logger.warning(f"Could not save static images for {gene1} vs {gene2}: {e}")

    logger.info(f"Created gene pair plots for {len(pairs_to_plot)} pairs")


def create_pca_plot(
    mutations_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Generate PCA plot from the per-sample gene VAF matrix.

    PCA is computed on a samples-by-genes matrix of maximum VAF values
    (standardized with z-scores).  Clinical data is NOT used as PCA input
    features; it is only used to color points by MPN subtype (PMF, ET, PV,
    etc.) when available.  If no clinical data is provided, points are
    colored by mutation count instead.

    Args:
        mutations_df: DataFrame with consolidated mutations (must contain
            sample_ID, gene, and a VAF column)
        clinical_df: DataFrame with clinical data (used only for MPN-type
            coloring; may be empty)
        output_dir: Directory to save plots
    """
    try:
        import plotly.express as px
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
    except ImportError:
        logger.error("Required packages not installed (plotly, scikit-learn)")
        return

    df = mutations_df.copy()
    df['sample_normalized'] = df['sample_ID'].apply(normalize_sample_id)

    vaf_col = 'SSCS.allele.frequency' if 'SSCS.allele.frequency' in df.columns else 'allele.frequency'

    # Build VAF matrix (samples x genes)
    samples = df['sample_normalized'].unique()
    genes = df['gene'].unique()

    # BUG-009: Use pivot_table instead of iterrows for vectorized construction
    df[vaf_col] = pd.to_numeric(df[vaf_col], errors='coerce').fillna(0)
    vaf_matrix = df.pivot_table(
        index='sample_normalized', columns='gene', values=vaf_col,
        aggfunc='max', fill_value=0.0
    ).reindex(index=samples, columns=genes, fill_value=0.0)

    # Extract MPN type for coloring (not used in PCA, just for visualization)
    mpn_types = pd.Series(index=samples, dtype=str)

    if not clinical_df.empty:
        clinical_dict = {}
        for _, row in clinical_df.iterrows():
            d_num = str(row.get('D#', '')).strip()
            if d_num:
                clinical_dict[d_num] = row

        for sample in samples:
            if sample in clinical_dict:
                mpn = str(clinical_dict[sample].get('MPN', '')).upper()
                if mpn and mpn not in ['NAN', 'NA', '', 'N', 'NO']:
                    mpn_types.loc[sample] = mpn
                else:
                    mpn_types.loc[sample] = 'None'
            else:
                mpn_types.loc[sample] = 'None'

    # Use VAF matrix only for PCA (no clinical features)
    feature_matrix = vaf_matrix.copy()
    feature_matrix = feature_matrix.fillna(0)

    # Remove samples with all zeros
    feature_matrix = feature_matrix[(feature_matrix != 0).any(axis=1)]

    if len(feature_matrix) < 3:
        logger.warning("Not enough samples for PCA")
        return

    # BUG-006: Bound n_components by both n_features and n_samples
    n_samples = len(feature_matrix)
    n_features = len(feature_matrix.columns)
    n_components = min(2, n_features, n_samples)

    if n_components < 2:
        logger.warning(
            f"Cannot run PCA: need at least 2 components but only "
            f"{n_components} possible (n_samples={n_samples}, n_features={n_features})"
        )
        return

    # Standardize and run PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    # RPR-005: Fix random_state for reproducible PCA results
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(
        pca_result[:, :2],
        columns=['PC1', 'PC2'],
        index=feature_matrix.index
    )
    pca_df['sample'] = pca_df.index

    # Add coloring variable (mutation count or MPN type)
    pca_df['n_mutations'] = (vaf_matrix.loc[pca_df.index] > 0).sum(axis=1)

    # Use MPN_type for coloring (categorical: PMF, ET, PV, etc.)
    if mpn_types.notna().any():
        pca_df['MPN'] = mpn_types.loc[pca_df.index].fillna('None')
        color_var = 'MPN'
    else:
        color_var = 'n_mutations'

    # Create plot
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=color_var,
        hover_data=['sample', 'n_mutations'],
        title='PCA: Mutation VAF Matrix'
    )

    fig.update_traces(marker=dict(size=12))

    # Add variance explained
    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        xaxis_title=f'PC1 ({var_explained[0]*100:.1f}% variance)',
        yaxis_title=f'PC2 ({var_explained[1]*100:.1f}% variance)',
        width=800,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    base = 'pca_clinical_mutations'

    fig.write_html(os.path.join(output_dir, f'{base}.html'))

    try:
        fig.write_image(os.path.join(output_dir, f'{base}.pdf'), format='pdf')
        fig.write_image(os.path.join(output_dir, f'{base}.png'), format='png', scale=2)
    except Exception as e:
        logger.warning(f"Could not save static PCA images: {e}")

    logger.info(f"Created PCA plot with {len(pca_df)} samples")


def generate_all_visualizations(
    mutations_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    base_output_dir: str
) -> None:
    """
    Generate all visualizations for Master_Output.

    Args:
        mutations_df: DataFrame with consolidated mutations
        clinical_df: DataFrame with clinical linkage data
        base_output_dir: Base directory for all outputs
    """
    logger.info("Starting visualization generation...")

    # 1. Oncoplot
    logger.info("Generating oncoplot...")
    create_oncoplot(
        mutations_df,
        os.path.join(base_output_dir, 'oncoplots')
    )

    # 2. VAF vs Age plots
    logger.info("Generating VAF vs Age plots...")
    create_vaf_vs_age_plots(
        mutations_df,
        clinical_df,
        os.path.join(base_output_dir, 'regression_plots', 'vaf_vs_age')
    )

    # 3. Gene co-occurrence analysis
    logger.info("Computing gene co-occurrence...")
    stats_dir = os.path.join(os.path.dirname(base_output_dir), 'stats')
    significant_pairs = compute_fisher_cooccurrence(
        mutations_df,
        stats_dir
    )

    # 4. Gene pair plots
    logger.info("Generating gene pair plots...")
    create_gene_pair_plots(
        mutations_df,
        significant_pairs,
        os.path.join(base_output_dir, 'regression_plots', 'gene_pairs')
    )

    # 5. PCA plot
    logger.info("Generating PCA plot...")
    create_pca_plot(
        mutations_df,
        clinical_df,
        os.path.join(base_output_dir, 'pca_plots')
    )

    logger.info("Visualization generation complete!")


if __name__ == '__main__':
    # Example: generate all visualizations from synthetic data.
    # Usage: SMMIP_ROOT=/path/to/data python visualization_generator.py
    logging.basicConfig(level=logging.INFO)

    smmip_root = os.environ.get("SMMIP_ROOT", ".")
    output_dir = os.path.join(smmip_root, 'Master_Output', 'output')

    # Minimal synthetic dataset for demonstration
    demo_mutations = pd.DataFrame({
        'batch_id': ['KG001'] * 6,
        'sample_ID': ['D1A,D1B', 'D2A', 'D3B', 'D4A,D4B', 'D5A', 'D6A'],
        'gene': ['DNMT3A', 'DNMT3A', 'TET2', 'TET2', 'JAK2', 'TP53'],
        'variant_type': ['missense_variant'] * 6,
        'SSCS.allele.frequency': [0.05, 0.12, 0.08, 0.15, 0.45, 0.04],
        'P-value': [0.01] * 6,
        'flags': [''] * 6,
    })
    demo_clinical = pd.DataFrame({
        'D#': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
        'Age': [65, 72, 58, 81, 45, 67],
        'Sex': ['M', 'F', 'M', 'F', 'M', 'F'],
        'MPN': ['PMF', '', '', 'ET', 'PV', ''],
    })

    generate_all_visualizations(demo_mutations, demo_clinical, output_dir)
    print(f"Visualizations saved to {output_dir}")
