"""
Convert MuData (RNA+ATAC in separate modalities) to AnnData (combined)
======================================================================
Converts ad_labeled_RNA_ATAC.h5mu to ad_labeled_RNA_ATAC.h5ad
for supervised classification with scVI.

This combines RNA and ATAC into a single matrix with:
- var['modality'] column indicating 'RNA' or 'ATAC'
- layers['counts'] with raw counts for both modalities
- .X with normalized data (for PCA fallback)
"""

import muon as mu
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

# Setup logging
os.makedirs('Logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/step2_3_convert_to_anndata.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("CONVERTING MUDATA TO ANNDATA FOR CLASSIFICATION")
    logger.info("="*80)
    
    INPUT_PATH = 'Data/processed/AD/ad_labeled_RNA_ATAC.h5mu'
    OUTPUT_PATH = 'Data/processed/AD/ad_mofa_annotated_RNA_ATAC.h5ad'
    
    try:
        # Load MuData
        logger.info(f"\nLoading MuData: {INPUT_PATH}")
        mdata = mu.read(INPUT_PATH)
        logger.info(f"  RNA: {mdata['rna'].shape}")
        logger.info(f"  ATAC: {mdata['atac'].shape}")
        
        # Get RNA and ATAC data
        rna = mdata['rna'].copy()
        atac = mdata['atac'].copy()
        
        # Store raw counts before normalization
        logger.info("\nStoring raw counts...")
        rna.layers['counts'] = rna.X.copy()
        atac.layers['counts'] = atac.X.copy()
        
        # Normalize both modalities
        logger.info("Normalizing RNA...")
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        
        logger.info("Normalizing ATAC...")
        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)
        
        # Combine into single AnnData
        logger.info("\nCombining RNA and ATAC into single AnnData...")
        
        # Use sparse matrices to save memory
        from scipy import sparse
        
        # Convert to sparse if not already
        if not sparse.issparse(rna.X):
            rna.X = sparse.csr_matrix(rna.X)
        if not sparse.issparse(atac.X):
            atac.X = sparse.csr_matrix(atac.X)
        if not sparse.issparse(rna.layers['counts']):
            rna.layers['counts'] = sparse.csr_matrix(rna.layers['counts'])
        if not sparse.issparse(atac.layers['counts']):
            atac.layers['counts'] = sparse.csr_matrix(atac.layers['counts'])
        
        logger.info("  Horizontally stacking sparse matrices...")
        # Create combined X matrix (normalized) - sparse
        X_combined = sparse.hstack([rna.X, atac.X], format='csr')
        
        # Create combined counts matrix (raw) - sparse
        counts_combined = sparse.hstack([rna.layers['counts'], atac.layers['counts']], format='csr')
        
        # Combine var - keep only essential columns
        logger.info("  Preparing var DataFrames...")
        
        # For RNA: just keep gene names and mark as RNA
        rna_var_simple = pd.DataFrame(index=rna.var.index)
        rna_var_simple['modality'] = 'RNA'
        rna_var_simple['feature_name'] = rna.var.index.values
        
        # For ATAC: keep peak names and mark as ATAC
        atac_var_simple = pd.DataFrame(index=atac.var.index)
        atac_var_simple['modality'] = 'ATAC'
        atac_var_simple['feature_name'] = atac.var.index.values
        
        var_combined = pd.concat([rna_var_simple, atac_var_simple], axis=0)
        
        logger.info(f"    RNA var: {len(rna_var_simple)} features")
        logger.info(f"    ATAC var: {len(atac_var_simple)} features")
        
        # Create combined AnnData
        logger.info("  Creating AnnData object...")
        adata_combined = ad.AnnData(
            X=X_combined,
            obs=rna.obs.copy(),  # Both have same obs
            var=var_combined
        )
        
        # Add raw counts layer
        adata_combined.layers['counts'] = counts_combined
        
        logger.info(f"  Combined shape: {adata_combined.shape}")
        logger.info(f"  RNA features: {(var_combined['modality'] == 'RNA').sum()}")
        logger.info(f"  ATAC features: {(var_combined['modality'] == 'ATAC').sum()}")
        logger.info(f"  Cell types: {adata_combined.obs['cell_type'].nunique()}")
        
        # Verify
        logger.info("\nVerifying data integrity...")
        logger.info(f"  .X shape: {adata_combined.X.shape}")
        logger.info(f"  .X dtype: {adata_combined.X.dtype}")
        logger.info(f"  .X format: {type(adata_combined.X)}")
        logger.info(f"  .layers['counts'] shape: {adata_combined.layers['counts'].shape}")
        logger.info(f"  .layers['counts'] dtype: {adata_combined.layers['counts'].dtype}")
        logger.info(f"  .obs shape: {adata_combined.obs.shape}")
        logger.info(f"  .var shape: {adata_combined.var.shape}")
        
        # For sparse matrices, check differently
        logger.info("  Checking for invalid values...")
        if sparse.issparse(adata_combined.X):
            logger.info("  .X is sparse - checking data array only")
            if np.any(np.isnan(adata_combined.X.data)):
                logger.warning("⚠️  NaN values found in .X")
            if np.any(np.isinf(adata_combined.X.data)):
                logger.warning("⚠️  Inf values found in .X")
        if sparse.issparse(adata_combined.layers['counts']):
            logger.info("  counts is sparse - checking data array only")
            if np.any(np.isnan(adata_combined.layers['counts'].data)):
                logger.warning("⚠️  NaN values found in counts")
            if np.any(adata_combined.layers['counts'].data < 0):
                logger.warning("⚠️  Negative values found in counts")
        
        # Save
        logger.info(f"\nSaving combined AnnData: {OUTPUT_PATH}")
        adata_combined.write_h5ad(OUTPUT_PATH)
        
        file_size = os.path.getsize(OUTPUT_PATH) / (1024**2)
        logger.info(f"  ✅ Saved: {file_size:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("✅ CONVERSION COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"Ready for classification!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
