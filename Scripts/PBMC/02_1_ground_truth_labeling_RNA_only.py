"""
FYDP Improved Method - Step 2.1: Ground Truth Labeling (RNA ONLY)
==================================================================
Generate ground truth labels using RNA-only data with MOFA+ and CellTypist.

Pipeline:
1. Load filtered data (use RNA only)
2. Z-score normalization
3. MOFA+ integration on RNA (20 factors)
4. Leiden clustering
5. CellTypist automated annotation

Input: Data/processed/pbmc_filtered.h5mu
Output: Data/processed/PBMC/pbmc_mofa_annotated_RNA_only.h5ad
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
os.makedirs('Data/outputs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/step2_1_RNA_only.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_mofa_rna_only(adata, n_factors=20):
    """Run MOFA+ integration on RNA only"""
    logger.info(f"  Running MOFA+ on RNA only ({n_factors} factors)...")
    
    try:
        from mofapy2.run.entry_point import entry_point
        
        # Prepare RNA data
        rna_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        
        logger.info(f"  RNA data: {rna_data.shape}")
        
        # Create nested list: [views][groups]
        # Single view (RNA), single group
        data = [[rna_data]]
        
        # Initialize MOFA
        ent = entry_point()
        
        # Set data options
        ent.set_data_options(
            scale_groups=False,
            scale_views=True
        )
        
        # Set data with matrix format
        ent.set_data_matrix(
            data,
            likelihoods=['gaussian'],
            views_names=['RNA'],
            groups_names=['group0'],
            samples_names=[adata.obs_names.tolist()],
            features_names=[adata.var_names.tolist()]
        )
        
        # Set model options
        ent.set_model_options(
            factors=n_factors,
            spikeslab_weights=True,
            ard_weights=True
        )
        
        # Set training options
        ent.set_train_options(
            iter=1000,
            convergence_mode='fast',
            verbose=False,
            seed=42
        )
        
        # Build and run
        ent.build()
        ent.run()
        
        # Get factors
        expectations = ent.model.getExpectations()
        factors = expectations['Z']['E']
        
        logger.info(f"  ✅ MOFA+ complete: {factors.shape}")
        return factors
        
    except ImportError:
        logger.warning("  ⚠️  MOFA+ not available, using PCA fallback")
        
        from sklearn.decomposition import PCA
        
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        pca = PCA(n_components=n_factors, random_state=42)
        factors = pca.fit_transform(X)
        
        logger.info(f"  ✅ PCA fallback complete: {factors.shape}")
        return factors

def run_celltypist_annotation(adata):
    """Run CellTypist automated annotation"""
    logger.info("  Running CellTypist annotation...")
    
    try:
        import celltypist
        from celltypist import models
        
        # Download and use Immune_All_Low model for PBMC
        logger.info("  Downloading CellTypist model...")
        models.download_models(force_update=False)
        
        # Run prediction
        logger.info("  Predicting cell types...")
        predictions = celltypist.annotate(
            adata,
            model='Immune_All_Low.pkl',
            majority_voting=True
        )
        
        # Add predictions to adata
        if hasattr(predictions, 'predicted_labels'):
            if isinstance(predictions.predicted_labels, pd.DataFrame):
                if 'majority_voting' in predictions.predicted_labels.columns:
                    adata.obs['celltypist_labels'] = predictions.predicted_labels['majority_voting'].values
                else:
                    adata.obs['celltypist_labels'] = predictions.predicted_labels['predicted_labels'].values
            else:
                adata.obs['celltypist_labels'] = predictions.predicted_labels
        
        if hasattr(predictions, 'conf_score'):
            adata.obs['celltypist_conf'] = predictions.conf_score
        
        logger.info(f"  ✅ CellTypist complete")
        logger.info(f"  Found {adata.obs['celltypist_labels'].nunique()} cell types")
        
        return adata
        
    except ImportError:
        logger.warning("  ⚠️  CellTypist not available, using Leiden clustering")
        
        sc.tl.leiden(adata, resolution=0.5, random_state=42)
        sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')
        
        adata.obs['celltypist_labels'] = 'Cluster_' + adata.obs['leiden'].astype(str)
        adata.obs['celltypist_conf'] = 1.0
        
        logger.info(f"  ✅ Leiden clustering complete: {adata.obs['leiden'].nunique()} clusters")
        
        return adata

def main():
    start_time = datetime.now()
    logger.info("\n" + "="*80)
    logger.info("STEP 2.1: GROUND TRUTH LABELING (RNA ONLY)")
    logger.info("="*80)
    
    # Paths
    INPUT_PATH = 'Data/processed/PBMC/pbmc_filtered.h5mu'
    OUTPUT_PATH = 'Data/processed/PBMC/pbmc_mofa_annotated_RNA_only.h5ad'
    MOFA_FACTORS_PATH = 'Results/Embeddings/mofa_factors_RNA_only.csv'
    
    # Parameters
    N_FACTORS = 20
    LEIDEN_RESOLUTION = 0.5
    
    try:
        # Load filtered data
        logger.info(f"\nLoading data from: {INPUT_PATH}")
        mdata = mu.read_h5mu(INPUT_PATH)
        logger.info(f"  RNA: {mdata['rna'].shape}")
        
        # Extract RNA only
        adata = mdata['rna'].copy()
        
        # Step 1: Save raw counts and normalize
        logger.info("\n" + "-"*80)
        logger.info("Step 2.1.1: Save Raw Counts & Normalize")
        logger.info("-"*80)
        
        logger.info("  Saving raw counts to layers['counts'] before normalization...")
        adata.layers['counts'] = adata.X.copy()
        
        # Normalize RNA
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata.copy()
        
        logger.info("  ✅ RNA normalized (ALL genes for CellTypist)")
        logger.info(f"     Genes: {adata.n_vars}")
        logger.info(f"     Raw counts preserved in layers['counts']")
        
        # Step 1.5: CellTypist Annotation (BEFORE feature selection)
        # CRITICAL: CellTypist must run on ALL genes, not just 2000 HVGs
        logger.info("\n" + "-"*80)
        logger.info("Step 2.1.1.5: CellTypist Annotation (BEFORE feature selection)")
        logger.info("-"*80)
        logger.info("  🔬 Running CellTypist on ALL genes before feature selection...")
        logger.info(f"  Input: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Create a copy with all genes for annotation
        adata_full_genes = adata.copy()
        adata_full_genes = run_celltypist_annotation(adata_full_genes)
        
        # Store the labels for later
        celltypist_labels = adata_full_genes.obs['celltypist_labels'].copy()
        celltypist_conf = adata_full_genes.obs.get('celltypist_conf', pd.Series(1.0, index=adata_full_genes.obs.index))
        
        logger.info(f"  ✅ CellTypist annotation complete on ALL {adata.n_vars} genes")
        logger.info(f"     Found {celltypist_labels.nunique()} cell types")
        
        # Now select highly variable genes for MOFA+
        logger.info("\n  Now selecting top 2000 variable genes for MOFA+ training...")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var.highly_variable].copy()
        
        # Z-score normalize
        sc.pp.scale(adata)
        
        logger.info(f"  ✅ Feature selection complete: {adata.n_vars} genes for MOFA+")
        
        # Step 2: MOFA+ Integration (RNA only)
        logger.info("\n" + "-"*80)
        logger.info("Step 2.1.2: MOFA+ Integration (RNA Only)")
        logger.info("-"*80)
        
        mofa_factors = run_mofa_rna_only(adata, n_factors=N_FACTORS)
        adata.obsm['X_mofa'] = mofa_factors
        
        # Save MOFA factors
        os.makedirs('Results/Embeddings', exist_ok=True)
        mofa_df = pd.DataFrame(
            mofa_factors,
            index=adata.obs_names,
            columns=[f'Factor_{i+1}' for i in range(N_FACTORS)]
        )
        mofa_df.to_csv(MOFA_FACTORS_PATH)
        logger.info(f"  ✅ Saved MOFA factors: {MOFA_FACTORS_PATH}")
        
        # Step 3: Leiden Clustering
        logger.info("\n" + "-"*80)
        logger.info("Step 2.1.3: Leiden Clustering")
        logger.info("-"*80)
        
        sc.pp.neighbors(adata, use_rep='X_mofa', n_neighbors=15, random_state=42)
        sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, random_state=42)
        sc.tl.umap(adata, random_state=42)
        
        logger.info(f"  ✅ Clustering complete: {adata.obs['leiden'].nunique()} clusters")
        
        # Step 4: Transfer CellTypist Annotations (from full gene annotation)
        logger.info("\n" + "-"*80)
        logger.info("Step 2.1.4: Transfer CellTypist Annotations")
        logger.info("-"*80)
        logger.info("  Transferring labels from full gene annotation...")
        
        # Transfer the pre-computed labels
        adata.obs['celltypist_labels'] = celltypist_labels
        adata.obs['celltypist_conf'] = celltypist_conf
        
        # Print summary
        logger.info("\n  Annotation Summary:")
        cell_type_counts = adata.obs['celltypist_labels'].value_counts()
        for cell_type, count in cell_type_counts.head(10).items():
            logger.info(f"    {cell_type:30s}: {count:5d} cells")
        
        # Save annotation summary
        summary_df = pd.DataFrame({
            'cell_type': cell_type_counts.index,
            'count': cell_type_counts.values,
            'percentage': (cell_type_counts.values / adata.n_obs * 100).round(2)
        })
        summary_path = 'Data/outputs/ground_truth_summary_RNA_only.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n  ✅ Saved annotation summary: {summary_path}")
        
        # Save annotated data
        logger.info(f"\nSaving annotated data to: {OUTPUT_PATH}")
        os.makedirs('Data/processed/PBMC', exist_ok=True)
        adata.write(OUTPUT_PATH)
        logger.info(f"  ✅ Saved: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("STEP 2.1 (RNA ONLY) COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Cells annotated: {adata.n_obs}")
    logger.info(f"Genes: {adata.n_vars}")
    logger.info(f"Cell types found: {adata.obs['celltypist_labels'].nunique()}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
