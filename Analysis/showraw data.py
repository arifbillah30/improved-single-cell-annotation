"""
Example: How to Read AD H5 File
================================
This script shows how to load and explore the AD multiome H5 file.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os

# =============================================================================
# METHOD 1: Basic Loading
# =============================================================================

# Get the script directory and go up one level to FYDP_Improved_Method
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

# Path to H5 file (relative to project root)
h5_path = os.path.join(project_dir, "Data/raw/AD/GSE214979_filtered_feature_bc_matrix.h5")

print(f"Looking for file at: {h5_path}")
print(f"File exists: {os.path.exists(h5_path)}\n")

# Load the file (gex_only=False includes both RNA and ATAC)
adata = sc.read_10x_h5(h5_path, gex_only=False)
adata.var_names_make_unique()

print(f"Loaded data: {adata.shape}")
print(f"  - {adata.n_obs:,} cells")
print(f"  - {adata.n_vars:,} features total")
print()

# Check what modalities are present
print("Feature types:")
print(adata.var['feature_types'].value_counts())
print()

# =============================================================================
# METHOD 2: Split into RNA and ATAC
# =============================================================================

# Create boolean masks
is_gene = adata.var['feature_types'] == 'Gene Expression'
is_peak = adata.var['feature_types'] == 'Peaks'

# Split into separate AnnData objects
adata_rna = adata[:, is_gene].copy()
adata_atac = adata[:, is_peak].copy()

print(f"RNA modality: {adata_rna.shape}")
print(f"  - {adata_rna.n_obs:,} cells")
print(f"  - {adata_rna.n_vars:,} genes")
print()

print(f"ATAC modality: {adata_atac.shape}")
print(f"  - {adata_atac.n_obs:,} cells")
print(f"  - {adata_atac.n_vars:,} peaks")
print()

# =============================================================================
# METHOD 3: Access Data
# =============================================================================

# The data is stored as a sparse matrix (to save memory)
print(f"Data type: {type(adata_rna.X)}")
print(f"Is sparse: {hasattr(adata_rna.X, 'toarray')}")
print()

# Convert to dense array (WARNING: Uses a lot of memory!)
# dense_data = adata_rna.X.toarray()

# Better: Access specific cells/genes
first_cell = adata_rna[0, :].X  # First cell, all genes
first_gene = adata_rna[:, 0].X  # All cells, first gene

# Get a subset (first 10 cells, first 10 genes)
subset = adata_rna[:10, :10].X.toarray()
print("First 10 cells × 10 genes:")
print(subset)
print()

# =============================================================================
# METHOD 4: Calculate Statistics
# =============================================================================

# Total counts per cell
if hasattr(adata_rna.X, 'toarray'):
    total_counts = np.array(adata_rna.X.sum(axis=1)).flatten()
    n_genes_detected = np.array((adata_rna.X > 0).sum(axis=1)).flatten()
else:
    total_counts = adata_rna.X.sum(axis=1)
    n_genes_detected = (adata_rna.X > 0).sum(axis=1)

print("RNA Statistics:")
print(f"  Total counts per cell:")
print(f"    - Min: {total_counts.min():,.0f}")
print(f"    - Median: {np.median(total_counts):,.0f}")
print(f"    - Max: {total_counts.max():,.0f}")
print()

print(f"  Genes detected per cell:")
print(f"    - Min: {n_genes_detected.min():.0f}")
print(f"    - Median: {np.median(n_genes_detected):.0f}")
print(f"    - Max: {n_genes_detected.max():.0f}")
print()

# =============================================================================
# METHOD 5: Explore Gene/Peak Names
# =============================================================================

print("First 10 genes:")
print(adata_rna.var_names[:10].tolist())
print()

print("First 10 ATAC peaks:")
print(adata_atac.var_names[:10].tolist())
print()

print("First 10 cell barcodes:")
print(adata_rna.obs_names[:10].tolist())
print()

# =============================================================================
# METHOD 6: Create DataFrame for Easier Viewing
# =============================================================================

# Convert a small subset to DataFrame
n_cells_to_show = 5
n_genes_to_show = 10

subset_data = adata_rna[:n_cells_to_show, :n_genes_to_show].X.toarray()
df = pd.DataFrame(
    subset_data,
    index=adata_rna.obs_names[:n_cells_to_show],
    columns=adata_rna.var_names[:n_genes_to_show]
)

print("Sample data as DataFrame:")
print(df)
print()

# =============================================================================
# METHOD 7: Save/Load Processed Data
# =============================================================================

# Save to H5AD format (scanpy's format)
# adata_rna.write('output_rna.h5ad')

# Load H5AD format
# adata_loaded = sc.read_h5ad('output_rna.h5ad')

print("="*80)
print("Done! Now you know how to read the H5 file.")
print("="*80)
