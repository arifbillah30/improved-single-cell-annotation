"""
FYDP Improved Method - Step 2.1: Ground Truth Labeling (AD RNA-only)
=====================================================================
Generate ground truth labels for AD neuronal cells using RNA-only data.

Unlike PBMC where we use CellTypist, for AD we use the pre-existing annotations
from the published GSE214979 dataset (Morabito et al., 2021).

Input: Data/processed/AD/ad_filtered.h5mu (QC filtered)
Output: Data/processed/AD/ad_labeled_RNA_only.h5mu (with ground truth labels)
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
        logging.FileHandler('Logs/step2_1_ad_rna_only_labeling.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_metadata_annotations():
    """Load cell type annotations from published dataset metadata"""
    logger.info("Loading published cell type annotations...")
    
    # Load metadata from GSE214979
    metadata_path = '/Users/arifbillah/Desktop/FYDP/Main/supervised-single-cell/Data/AD/GSE214979_cell_metadata.csv'
    
    try:
        metadata = pd.read_csv(metadata_path, index_col=0)
        logger.info(f"✅ Loaded metadata: {metadata.shape}")
        logger.info(f"   Available cell types: {metadata['predicted.id'].unique()}")
        logger.info(f"   Available subtypes: {metadata['subs'].unique()}")
        
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        raise

def main():
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("FYDP IMPROVED METHOD - STEP 2.1: GROUND TRUTH LABELING (AD RNA-ONLY)")
    logger.info("="*80)
    logger.info("Using published annotations from GSE214979")
    logger.info("")
    
    # Paths
    INPUT_PATH = 'Data/processed/AD/ad_filtered.h5mu'
    OUTPUT_PATH = 'Data/processed/AD/ad_labeled_RNA_only.h5mu'
    
    try:
        # Load filtered data
        logger.info(f"Loading filtered data: {INPUT_PATH}")
        mdata = mu.read(INPUT_PATH)
        adata = mdata['rna']
        
        logger.info(f"✅ Loaded data: {adata.shape}")
        
        # Load metadata annotations
        metadata = load_metadata_annotations()
        
        # Match cells
        common_cells = adata.obs_names.intersection(metadata.index)
        logger.info(f"\nMatching cells with metadata:")
        logger.info(f"  Cells in filtered data: {adata.n_obs:,}")
        logger.info(f"  Cells in metadata: {len(metadata):,}")
        logger.info(f"  Common cells: {len(common_cells):,} ({100*len(common_cells)/adata.n_obs:.1f}%)")
        
        # Filter to common cells
        adata = adata[common_cells, :].copy()
        metadata_filtered = metadata.loc[common_cells]
        
        # Add annotations
        logger.info("\n" + "="*80)
        logger.info("ADDING CELL TYPE ANNOTATIONS")
        logger.info("="*80)
        
        # Major cell types (predicted.id column)
        adata.obs['cell_type'] = metadata_filtered['predicted.id'].values
        adata.obs['cell_subtype'] = metadata_filtered['subs'].values
        
        # Add condition (AD vs Control)
        adata.obs['diagnosis'] = metadata_filtered['Diagnosis'].values
        adata.obs['braak_stage'] = metadata_filtered['Braak'].values
        
        logger.info(f"Cell type distribution:")
        type_counts = adata.obs['cell_type'].value_counts()
        for cell_type, count in type_counts.items():
            logger.info(f"  {cell_type}: {count:,} ({100*count/len(adata):.1f}%)")
        
        logger.info(f"\nCell subtype distribution:")
        subtype_counts = adata.obs['cell_subtype'].value_counts()
        for subtype, count in list(subtype_counts.items())[:10]:
            logger.info(f"  {subtype}: {count:,} ({100*count/len(adata):.1f}%)")
        if len(subtype_counts) > 10:
            logger.info(f"  ... and {len(subtype_counts) - 10} more subtypes")
        
        logger.info(f"\nDiagnosis distribution:")
        diag_counts = adata.obs['diagnosis'].value_counts()
        for diag, count in diag_counts.items():
            logger.info(f"  {diag}: {count:,} ({100*count/len(adata):.1f}%)")
        
        # Basic preprocessing for embeddings
        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING FOR VISUALIZATION")
        logger.info("="*80)
        
        logger.info("Normalizing and log-transforming...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        logger.info("Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
        logger.info(f"  Found {adata.var['highly_variable'].sum()} HVGs")
        
        logger.info("Computing PCA...")
        sc.pp.scale(adata)
        sc.tl.pca(adata, n_comps=50)
        
        logger.info("Computing UMAP...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata)
        
        # Save
        logger.info("\n" + "="*80)
        logger.info("SAVING LABELED DATA")
        logger.info("="*80)
        
        # Update MuData - need to update modality in place
        mdata.mod['rna'] = adata
        mdata.update()
        mdata.write(OUTPUT_PATH)
        
        logger.info(f"✅ Saved labeled data: {OUTPUT_PATH}")
        logger.info(f"   RNA: {adata.shape}")
        logger.info(f"   Size: {os.path.getsize(OUTPUT_PATH) / (1024**2):.1f} MB")
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.error("\nPlease ensure you have:")
        logger.error(f"  1. Run Step 1 to generate: {INPUT_PATH}")
        logger.error(f"  2. Metadata file at: ../../Data/AD/GSE214979_cell_metadata.csv")
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
    logger.info("✅ STEP 2.1 COMPLETE: GROUND TRUTH LABELING (AD RNA-ONLY)")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"Cell types: {adata.obs['cell_type'].nunique()}")
    logger.info(f"Cell subtypes: {adata.obs['cell_subtype'].nunique()}")
    logger.info(f"")
    logger.info(f"Next step: Run Step 2.2 for RNA+ATAC labeling")
    logger.info("="*80)

if __name__ == "__main__":
    main()
