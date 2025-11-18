"""
FYDP Improved Method - Step 2.2: Ground Truth Labeling (AD RNA+ATAC)
======================================================================
Generate ground truth labels for AD neuronal cells using RNA+ATAC data.

Uses the same pre-existing annotations as RNA-only but keeps both modalities.

Input: Data/processed/AD/ad_filtered.h5mu (QC filtered with RNA + ATAC)
Output: Data/processed/AD/ad_labeled_RNA_ATAC.h5mu (with ground truth labels, both modalities)
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
        logging.FileHandler('Logs/step2_2_ad_rna_atac_labeling.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_metadata_annotations():
    """Load cell type annotations from published dataset metadata"""
    logger.info("Loading published cell type annotations...")
    
    metadata_path = '/Users/arifbillah/Desktop/FYDP/Main/supervised-single-cell/Data/AD/GSE214979_cell_metadata.csv'
    
    try:
        metadata = pd.read_csv(metadata_path, index_col=0)
        logger.info(f"✅ Loaded metadata: {metadata.shape}")
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        raise

def main():
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("FYDP IMPROVED METHOD - STEP 2.2: GROUND TRUTH LABELING (AD RNA+ATAC)")
    logger.info("="*80)
    logger.info("Using published annotations from GSE214979")
    logger.info("")
    
    # Paths
    INPUT_PATH = 'Data/processed/AD/ad_filtered.h5mu'
    OUTPUT_PATH = 'Data/processed/AD/ad_labeled_RNA_ATAC.h5mu'
    
    try:
        # Load filtered data
        logger.info(f"Loading filtered data: {INPUT_PATH}")
        mdata = mu.read(INPUT_PATH)
        
        logger.info(f"✅ Loaded data:")
        logger.info(f"   RNA: {mdata['rna'].shape}")
        logger.info(f"   ATAC: {mdata['atac'].shape}")
        
        # Load metadata annotations
        metadata = load_metadata_annotations()
        
        # Match cells
        common_cells = mdata['rna'].obs_names.intersection(metadata.index)
        logger.info(f"\nMatching cells with metadata:")
        logger.info(f"  Cells in filtered data: {mdata['rna'].n_obs:,}")
        logger.info(f"  Cells in metadata: {len(metadata):,}")
        logger.info(f"  Common cells: {len(common_cells):,} ({100*len(common_cells)/mdata['rna'].n_obs:.1f}%)")
        
        # Filter to common cells
        mdata = mdata[common_cells, :].copy()
        metadata_filtered = metadata.loc[common_cells]
        
        # Add annotations to RNA modality
        logger.info("\n" + "="*80)
        logger.info("ADDING CELL TYPE ANNOTATIONS")
        logger.info("="*80)
        
        mdata['rna'].obs['cell_type'] = metadata_filtered['predicted.id'].values
        mdata['rna'].obs['cell_subtype'] = metadata_filtered['subs'].values
        mdata['rna'].obs['diagnosis'] = metadata_filtered['Diagnosis'].values
        mdata['rna'].obs['braak_stage'] = metadata_filtered['Braak'].values
        
        # Also add to ATAC modality for consistency
        mdata['atac'].obs['cell_type'] = metadata_filtered['predicted.id'].values
        mdata['atac'].obs['cell_subtype'] = metadata_filtered['subs'].values
        mdata['atac'].obs['diagnosis'] = metadata_filtered['Diagnosis'].values
        
        logger.info(f"Cell type distribution:")
        type_counts = mdata['rna'].obs['cell_type'].value_counts()
        for cell_type, count in type_counts.items():
            logger.info(f"  {cell_type}: {count:,} ({100*count/len(mdata['rna']):.1f}%)")
        
        logger.info(f"\nDiagnosis distribution:")
        diag_counts = mdata['rna'].obs['diagnosis'].value_counts()
        for diag, count in diag_counts.items():
            logger.info(f"  {diag}: {count:,} ({100*count/len(mdata['rna']):.1f}%)")
        
        # Skip preprocessing to save memory - will be done in classification step
        logger.info("\n" + "="*80)
        logger.info("SKIPPING PREPROCESSING (will be done in classification step)")
        logger.info("="*80)
        logger.info("This saves memory and avoids duplication")
        logger.info("Classification scripts will handle normalization + embedding")
        
        # Save
        logger.info("\n" + "="*80)
        logger.info("SAVING LABELED DATA")
        logger.info("="*80)
        
        mdata.write(OUTPUT_PATH)
        
        logger.info(f"✅ Saved labeled data: {OUTPUT_PATH}")
        logger.info(f"   RNA: {mdata['rna'].shape}")
        logger.info(f"   ATAC: {mdata['atac'].shape}")
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
    logger.info("✅ STEP 2.2 COMPLETE: GROUND TRUTH LABELING (AD RNA+ATAC)")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"Cell types: {mdata['rna'].obs['cell_type'].nunique()}")
    logger.info(f"")
    logger.info(f"Next step: Run Step 3 classification experiments")
    logger.info("="*80)

if __name__ == "__main__":
    main()
