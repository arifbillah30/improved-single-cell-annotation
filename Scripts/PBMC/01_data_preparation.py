"""
FYDP Improved Method - Step 1: Data Preparation
================================================
Load PBMC 10K multiome dataset and perform QC filtering.

This is STEP 1 of our NEW methodology - we start from raw data
and do NOT use the old WNN-based ground truth annotations.

Input: Raw PBMC 10K .h5 file (contains both RNA and ATAC)
Output: Data/processed/pbmc_filtered.h5mu (QC filtered, NO annotations yet)

Based on Utils.py quality_control() function from old code
"""

import scanpy as sc
import muon as mu
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys

# Setup logging
os.makedirs('Logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/fydp_improved.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("FYDP IMPROVED METHOD - STEP 1: DATA PREPARATION")
    logger.info("="*80)
    logger.info("Starting from RAW data - no pre-existing annotations")
    logger.info("This matches the reference paper's starting point")
    logger.info("")
    
    # Configuration
    RAW_DATA_PATH = 'Data/raw/PBMC/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5'
    OUTPUT_PATH = 'Data/processed/PBMC/pbmc_filtered.h5mu'
    
    # QC thresholds (SAME as reference paper - from Utils.py quality_control)
    MIN_GENES = 500
    MAX_GENES = 5000
    MAX_MT_PERCENT = 20
    MAX_TOTAL_COUNTS = 15000
    MIN_UNIQUE_PEAKS = 2000
    MAX_UNIQUE_PEAKS = 15000
    MIN_TOTAL_PEAKS = 4000
    MAX_TOTAL_PEAKS = 40000
    
    logger.info(f"Loading raw data from: {RAW_DATA_PATH}")
    
    try:
        # Load h5 file (10x multiome format with RNA + ATAC combined)
        # IMPORTANT: Use gex_only=False to read BOTH RNA and ATAC
        adata_full = sc.read_10x_h5(RAW_DATA_PATH, gex_only=False)
        adata_full.var_names_make_unique()
        
        logger.info(f"✅ Loaded multiome data: {adata_full.shape}")
        logger.info(f"   Feature types: {adata_full.var['feature_types'].value_counts().to_dict()}")
        
        # Split into RNA and ATAC modalities based on feature_types column
        is_gene = adata_full.var['feature_types'] == 'Gene Expression'
        is_peak = adata_full.var['feature_types'] == 'Peaks'
        
        adata_rna = adata_full[:, is_gene].copy()
        adata_atac = adata_full[:, is_peak].copy()
        
        logger.info(f"   RNA modality: {adata_rna.shape}")
        logger.info(f"   ATAC modality: {adata_atac.shape}")
        
        # Store original counts
        n_cells_original = adata_rna.n_obs
        n_genes_original = adata_rna.n_vars
        n_peaks_original = adata_atac.n_vars
        
        logger.info("\n" + "="*80)
        logger.info("QC FILTERING - RNA (same thresholds as reference paper)")
        logger.info("="*80)
        
        # Calculate QC metrics for RNA (from Utils.py)
        adata_rna.var['mt'] = adata_rna.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            adata_rna, 
            qc_vars=['mt'], 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
        logger.info(f"Before filtering:")
        logger.info(f"  Cells: {adata_rna.n_obs:,}")
        logger.info(f"  Genes: {adata_rna.n_vars:,}")
        logger.info(f"  Median genes/cell: {np.median(adata_rna.obs['n_genes_by_counts']):.0f}")
        logger.info(f"  Median total counts: {np.median(adata_rna.obs['total_counts']):.0f}")
        logger.info(f"  Median MT%: {np.median(adata_rna.obs['pct_counts_mt']):.2f}%")
        
        # Filter cells by gene count (MIN_GENES <= n_genes < MAX_GENES)
        mu.pp.filter_obs(adata_rna, 'n_genes_by_counts', lambda x: (x >= MIN_GENES) & (x < MAX_GENES))
        logger.info(f"After gene count filter ({MIN_GENES}-{MAX_GENES}): {adata_rna.n_obs:,} cells")
        
        # Filter cells by total counts (remove doublets)
        mu.pp.filter_obs(adata_rna, 'total_counts', lambda x: x < MAX_TOTAL_COUNTS)
        logger.info(f"After total counts filter (<{MAX_TOTAL_COUNTS}): {adata_rna.n_obs:,} cells")
        
        # Filter cells by MT percentage
        mu.pp.filter_obs(adata_rna, 'pct_counts_mt', lambda x: x < MAX_MT_PERCENT)
        logger.info(f"After MT% filter (<{MAX_MT_PERCENT}%): {adata_rna.n_obs:,} cells")
        
        # Filter genes (keep genes expressed in at least 3 cells)
        sc.pp.filter_genes(adata_rna, min_cells=3)
        
        logger.info(f"\nAfter RNA filtering:")
        logger.info(f"  Cells: {adata_rna.n_obs:,}")
        logger.info(f"  Genes: {adata_rna.n_vars:,}")
        logger.info(f"  Removed: {n_cells_original - adata_rna.n_obs:,} cells ({100*(n_cells_original - adata_rna.n_obs)/n_cells_original:.1f}%)")
        
        logger.info("\n" + "="*80)
        logger.info("QC FILTERING - ATAC (same thresholds as reference paper)")
        logger.info("="*80)
        
        # Calculate QC metrics for ATAC
        sc.pp.calculate_qc_metrics(adata_atac, percent_top=None, log1p=False, inplace=True)
        
        logger.info(f"Before filtering:")
        logger.info(f"  Cells: {adata_atac.n_obs:,}")
        logger.info(f"  Peaks: {adata_atac.n_vars:,}")
        logger.info(f"  Median peaks/cell: {np.median(adata_atac.obs['n_genes_by_counts']):.0f}")
        logger.info(f"  Median total counts: {np.median(adata_atac.obs['total_counts']):.0f}")
        
        # Filter ATAC cells by unique peak counts
        mu.pp.filter_obs(adata_atac, 'n_genes_by_counts', 
                        lambda x: (x >= MIN_UNIQUE_PEAKS) & (x <= MAX_UNIQUE_PEAKS))
        logger.info(f"After unique peaks filter ({MIN_UNIQUE_PEAKS}-{MAX_UNIQUE_PEAKS}): {adata_atac.n_obs:,} cells")
        
        # Filter ATAC cells by total peak counts
        mu.pp.filter_obs(adata_atac, 'total_counts',
                        lambda x: (x >= MIN_TOTAL_PEAKS) & (x <= MAX_TOTAL_PEAKS))
        logger.info(f"After total peaks filter ({MIN_TOTAL_PEAKS}-{MAX_TOTAL_PEAKS}): {adata_atac.n_obs:,} cells")
        
        logger.info("\n" + "="*80)
        logger.info("MATCHING MODALITIES (keep cells passing QC in BOTH)")
        logger.info("="*80)
        
        # Keep only cells that passed QC in BOTH RNA and ATAC
        common_cells = adata_rna.obs_names.intersection(adata_atac.obs_names)
        logger.info(f"Common cells after QC: {len(common_cells):,}")
        
        adata_rna = adata_rna[common_cells, :].copy()
        adata_atac = adata_atac[common_cells, :].copy()
        
        logger.info(f"Final QC-filtered shapes:")
        logger.info(f"  RNA: {adata_rna.shape}")
        logger.info(f"  ATAC: {adata_atac.shape}")
        logger.info(f"  Total cells retained: {adata_rna.n_obs:,} ({100*adata_rna.n_obs/n_cells_original:.1f}%)")
        
        logger.info("\n" + "="*80)
        logger.info("SAVING QC-FILTERED DATA (NO ANNOTATIONS YET)")
        logger.info("="*80)
        logger.info("⚠️  Note: Cell type annotations will be generated in Step 2")
        logger.info("    using our NEW method (MOFA+ + CellTypist)")
        logger.info("")
        
        # Create MuData object
        mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})
        
        # Save
        os.makedirs('Data/processed/PBMC', exist_ok=True)
        mdata.write(OUTPUT_PATH)
        logger.info(f"✅ Saved QC-filtered MuData: {OUTPUT_PATH}")
        logger.info(f"   RNA: {adata_rna.shape}")
        logger.info(f"   ATAC: {adata_atac.shape}")
        logger.info(f"   Size: {os.path.getsize(OUTPUT_PATH) / (1024**2):.1f} MB")
        
    except FileNotFoundError:
        logger.error(f"❌ Data file not found: {RAW_DATA_PATH}")
        logger.error("\nPlease ensure you have copied the raw data to:")
        logger.error(f"   {RAW_DATA_PATH}")
        logger.error("\nFrom the old supervised-single-cell folder:")
        logger.error("   cp ../Data/PBMC/pbmc_filtered.h5 Data/raw/PBMC/")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("✅ STEP 1 COMPLETE: DATA PREPARATION")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"")
    logger.info(f"Comparison to reference paper:")
    logger.info(f"  Paper used: {n_cells_original:,} cells → QC → ~9,814 cells (82.4%)")
    logger.info(f"  We got: {n_cells_original:,} cells → QC → {adata_rna.n_obs:,} cells ({100*adata_rna.n_obs/n_cells_original:.1f}%)")
    logger.info(f"")
    logger.info(f"Next step: python Scripts/02_ground_truth_labeling.py")
    logger.info(f"  This will create NEW ground truth using MOFA+ + CellTypist")
    logger.info(f"  (NOT the old WNN-based annotations)")
    logger.info("="*80)

if __name__ == "__main__":
    main()
