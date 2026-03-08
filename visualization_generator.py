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
import logging
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Variant type color scheme (from CHIP_analysis.R)
VARIANT_COLORS = {
    'missense_variant': '#3498db',      # Blue
    'Missense': '#3498db',
    'stop_gained': '#e74c3c',           # Red
    'Nonsense': '#e74c3c',
    'frameshift_variant': '#8e44ad',    # Dark Purple
    'Deletion': '#8e44ad',
    'inframe_insertion': '#9b59b6',     # Purple
    'Insertion': '#9b59b6',
    'inframe_deletion': '#a569bd',      # Light Purple
    'Indel': '#a569bd',
    'splice_region_variant': '#f39c12', # Orange
    'splice_donor_variant': '#f39c12',
    'splice_acceptor_variant': '#f39c12',
    'Splice_Site': '#f39c12',
    'synonymous_variant': '#95a5a6',    # Gray
    'Synonymous': '#95a5a6',
    'intron_variant': '#7f8c8d',        # Dark Gray
    'Intronic': '#7f8c8d',
    '5_prime_UTR_variant': '#bdc3c7',   # Light Gray
    '3_prime_UTR_variant': '#bdc3c7',
    'Non_Coding': '#bdc3c7',
    'Other': '#2c3e50'                  # Very Dark Blue
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
    """Normalize variant type for consistent coloring."""
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


def normalize_sample_id(sample_id: str) -> str:
    """Normalize sample ID (remove A/B suffix)."""
    if pd.isna(sample_id):
        return ""
    base_id = str(sample_id).split(',')[0].strip()
    return re.sub(r'[AB]$', '', base_id)


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
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("Plotly not installed. Run: pip install plotly kaleido")
        return

    # NOTE: Filtering is handled by the manager (master_output_manager.py line 318)
    # via mutation_consolidator.filter_mutations() before data reaches this function.
    # Do NOT duplicate filtering here — the manager's pre-filter is authoritative.
    df = mutations_df.copy()

    if df.empty:
        logger.warning("No mutations pass filters for oncoplot")
        return

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

    # Create 2x2 subplot grid:
    # [top bar chart] [empty corner]
    # [main heatmap]  [right bar chart]
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.12, 0.88],
        column_widths=[0.85, 0.15],
        vertical_spacing=0.05,
        horizontal_spacing=0.02,
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],  # Top row
            [{"type": "heatmap"}, {"type": "bar"}]   # Bottom row
        ]
    )

    # =========================================================================
    # Top bar chart - mutations per sample (row 1, col 1)
    # =========================================================================
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

    # =========================================================================
    # Main heatmap (row 2, col 1)
    # =========================================================================
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

    # =========================================================================
    # Right bar chart - samples per gene (row 2, col 2)
    # =========================================================================
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

    # =========================================================================
    # Layout configuration
    # =========================================================================
    fig_height = max(600, 35 * len(genes) + 250)
    fig_width = max(1000, 18 * len(samples) + 250)

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

    for gene in genes:
        gene_df = df[df['gene'] == gene]

        if len(gene_df) < 3:
            continue

        # Skip if all ages are identical (can't do regression)
        if gene_df['Age'].nunique() < 2:
            logger.debug(f"Skipping {gene}: all ages identical")
            continue

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

        # Calculate regression stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            gene_df['Age'], gene_df['VAF']
        )

        # Add stats annotation
        fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=f'R² = {r_value**2:.3f}<br>p = {p_value:.3e}<br>n = {len(gene_df)}',
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

    logger.info(f"Created VAF vs Age plots for {len(genes)} genes")


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

    gene_sample_matrix = pd.DataFrame(0, index=samples, columns=genes)

    for _, row in df.iterrows():
        gene_sample_matrix.loc[row['sample_normalized'], row['gene']] = 1

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

        # Calculate correlation
        r, p = stats.pearsonr(merged['VAF1'], merged['VAF2'])

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
            text=f'r = {r:.3f}<br>p = {p:.3e}<br>n = {len(merged)}<br>Fisher p = {pair_row["p_value"]:.3e}',
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
    Generate PCA plot with clinical and mutation features.

    Args:
        mutations_df: DataFrame with consolidated mutations
        clinical_df: DataFrame with clinical data
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

    vaf_matrix = pd.DataFrame(0.0, index=samples, columns=genes)

    for _, row in df.iterrows():
        sample = row['sample_normalized']
        gene = row['gene']
        vaf = float(row[vaf_col]) if pd.notna(row[vaf_col]) else 0
        # Take max VAF if multiple mutations in same gene
        vaf_matrix.loc[sample, gene] = max(vaf_matrix.loc[sample, gene], vaf)

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

    # Standardize and run PCA
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    pca = PCA(n_components=min(3, len(feature_matrix.columns)))
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
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    print("Testing visualization generator...")

    # Create sample mutation data
    test_mutations = pd.DataFrame({
        'batch_id': ['KG001'] * 10,
        'sample_ID': ['D1A,D1B', 'D2A', 'D3B', 'D4A,D4B', 'D5A',
                      'D1A,D1B', 'D2A', 'D3B', 'D4A,D4B', 'D6A'],
        'gene': ['DNMT3A', 'DNMT3A', 'TET2', 'TET2', 'JAK2',
                 'TET2', 'ASXL1', 'ASXL1', 'SF3B1', 'TP53'],
        'variant_type': ['missense_variant'] * 10,
        'SSCS.allele.frequency': [0.05, 0.12, 0.08, 0.15, 0.45,
                                   0.03, 0.07, 0.11, 0.22, 0.04],
        'P-value': [0.01] * 10,
        'flags': [''] * 10
    })

    test_clinical = pd.DataFrame({
        'D#': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'],
        'Age': [65, 72, 58, 81, 45, 67],
        'Sex': ['M', 'F', 'M', 'F', 'M', 'F'],
        'MPN': ['PMF', '', '', 'ET', 'PV', '']
    })

    output_dir = '/Volumes/Seq_SSD/smMIP/Master_Output/output'

    try:
        generate_all_visualizations(test_mutations, test_clinical, output_dir)
        print("Test visualizations created successfully!")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
