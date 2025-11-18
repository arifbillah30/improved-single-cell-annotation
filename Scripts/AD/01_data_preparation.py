"""
FYDP Improved Method - Step 1: Data Preparation (AD Dataset)
=============================================================
Load AD (Alzheimer's Disease) neuronal 10x multiome dataset and perform QC filtering.

This is STEP 1 of our NEW methodology for AD neuronal cells - we start from raw data
and do NOT use pre-existing annotations.

Input: Raw AD .h5 file (contains both RNA and ATAC)
Output: Data/processed/AD/ad_filtered.h5mu (QC filtered, NO annotations yet)

Dataset: GSE214979 - 10,530 neuronal cells from 7 AD patients + 8 controls
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
        logging.FileHandler('Logs/step1_ad_data_prep.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("FYDP IMPROVED METHOD - STEP 1: DATA PREPARATION (AD NEURONAL CELLS)")
    logger.info("="*80)
    logger.info("Dataset: Alzheimer's Disease neuronal cells")
    logger.info("Starting from RAW data - no pre-existing annotations")
    logger.info("")
    
    # Configuration
    RAW_DATA_PATH = 'Data/raw/AD/GSE214979_filtered_feature_bc_matrix.h5'
    OUTPUT_PATH = 'Data/processed/AD/ad_filtered.h5mu'
    
    # QC thresholds (adjusted for neuronal cells - typically have fewer genes than immune cells)
    MIN_GENES = 200          # Lower than PBMC (500) - neurons have less transcription
    MAX_GENES = 6000         # Slightly higher than PBMC (5000)
    MAX_MT_PERCENT = 25      # Higher than PBMC (20) - neurons tolerate more MT
    MAX_TOTAL_COUNTS = 25000 # Higher than PBMC (15000)
    MIN_UNIQUE_PEAKS = 1000  # Lower than PBMC (2000) - neurons less accessible
    MAX_UNIQUE_PEAKS = 20000 # Higher than PBMC (15000)
    MIN_TOTAL_PEAKS = 2000   # Lower than PBMC (4000)
    MAX_TOTAL_PEAKS = 50000  # Higher than PBMC (40000)
    
    logger.info(f"Loading raw data from: {RAW_DATA_PATH}")
    
    try:
        # Load h5 file (10x multiome format with RNA + ATAC combined)
        adata_full = sc.read_10x_h5(RAW_DATA_PATH, gex_only=False)
        adata_full.var_names_make_unique()
        
        logger.info(f"✅ Loaded multiome data: {adata_full.shape}")
        logger.info(f"   Feature types: {adata_full.var['feature_types'].value_counts().to_dict()}")
        
        # Split into RNA and ATAC modalities
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
        logger.info("QC FILTERING - RNA (adjusted for neuronal cells)")
        logger.info("="*80)
        
        # Calculate QC metrics for RNA
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
        
        # Filter cells by gene count
        mu.pp.filter_obs(adata_rna, 'n_genes_by_counts', lambda x: (x >= MIN_GENES) & (x < MAX_GENES))
        logger.info(f"After gene count filter ({MIN_GENES}-{MAX_GENES}): {adata_rna.n_obs:,} cells")
        
        # Filter cells by total counts
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
        logger.info("QC FILTERING - ATAC (adjusted for neuronal cells)")
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
        logger.info("")
        
        # Create MuData object
        mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})
        
        # Save
        os.makedirs('Data/processed/AD', exist_ok=True)
        mdata.write(OUTPUT_PATH)
        logger.info(f"✅ Saved QC-filtered MuData: {OUTPUT_PATH}")
        logger.info(f"   RNA: {adata_rna.shape}")
        logger.info(f"   ATAC: {adata_atac.shape}")
        logger.info(f"   Size: {os.path.getsize(OUTPUT_PATH) / (1024**2):.1f} MB")
        
    except FileNotFoundError:
        logger.error(f"❌ Data file not found: {RAW_DATA_PATH}")
        logger.error("\nPlease ensure you have the raw data at:")
        logger.error(f"   {RAW_DATA_PATH}")
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
    logger.info("✅ STEP 1 COMPLETE: DATA PREPARATION (AD)")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"")
    logger.info(f"Comparison to reference paper:")
    logger.info(f"  Paper used: 10,530 neuronal cells from 7 AD + 8 control patients")
    logger.info(f"  We got: {n_cells_original:,} cells → QC → {adata_rna.n_obs:,} cells")
    logger.info(f"")
    logger.info(f"Next step: Run Step 2 (ground truth labeling) for AD dataset")
    logger.info("="*80)

if __name__ == "__main__":
    main()
