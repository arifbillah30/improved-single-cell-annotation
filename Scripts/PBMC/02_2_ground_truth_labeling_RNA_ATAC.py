"""
FYDP Improved Method - Step 2.2: Ground Truth Labeling (RNA + ATAC COMBINED)
=============================================================================
Generate ground truth labels using RNA+ATAC combined for scVI training.

Pipeline:
1. Load filtered data
2. Save raw counts for BOTH RNA and ATAC
3. Normalize and concatenate RNA + ATAC features
4. MOFA+ integration (20 factors)
5. Leiden clustering
6. CellTypist automated annotation
7. Save combined RNA+ATAC AnnData for scVI

Input: Data/processed/pbmc_filtered.h5mu
Output: Data/processed/PBMC/pbmc_mofa_annotated_RNA_ATAC.h5ad
"""

import scanpy as sc
import muon as mu
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys
from scipy import sparse

# Setup logging
os.makedirs('Logs', exist_ok=True)
os.makedirs('Data/outputs/RNA_ATAC', exist_ok=True)
os.makedirs('Results/Embeddings/RNA_ATAC', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/step2_2_RNA_ATAC.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def select_top_features(mdata, n_rna_features=2000, n_atac_features=5000):
    """Select top highly variable features"""
    logger.info(f"  Selecting top features: RNA={n_rna_features}, ATAC={n_atac_features}")
    
    # RNA: Use highly variable genes
    if 'highly_variable' in mdata['rna'].var.columns:
        hvg = mdata['rna'].var['highly_variable']
        n_hvg = hvg.sum()
        logger.info(f"    RNA: Using {n_hvg} highly variable genes")
        mdata.mod['rna'] = mdata.mod['rna'][:, hvg].copy()
    else:
        logger.warning(f"    RNA: highly_variable not found, using all {mdata['rna'].n_vars} genes")
    
    # ATAC: Select top variable peaks
    atac_data = mdata['atac'].X.toarray() if hasattr(mdata['atac'].X, 'toarray') else mdata['atac'].X
    atac_var = np.var(atac_data, axis=0)
    
    if len(atac_var) > n_atac_features:
        top_peaks_idx = np.argsort(atac_var)[-n_atac_features:]
        logger.info(f"    ATAC: Selecting {n_atac_features} most variable peaks (out of {len(atac_var)})")
        mdata.mod['atac'] = mdata.mod['atac'][:, top_peaks_idx].copy()
    else:
        logger.info(f"    ATAC: Using all {len(atac_var)} peaks")
    
    logger.info(f"  Feature selection complete: RNA={mdata['rna'].shape}, ATAC={mdata['atac'].shape}")
    return mdata

def remove_zero_variance_features(mdata, min_std=1e-5):
    """Remove features with zero or near-zero variance"""
    logger.info("  Removing zero-variance features...")
    
    # RNA
    rna_data = mdata['rna'].X.toarray() if hasattr(mdata['rna'].X, 'toarray') else mdata['rna'].X
    rna_std = np.std(rna_data, axis=0)
    rna_keep = rna_std > min_std
    n_rna_dropped = (~rna_keep).sum()
    
    if n_rna_dropped > 0:
        logger.info(f"    Dropping {n_rna_dropped} zero-variance RNA features")
        mdata.mod['rna'] = mdata.mod['rna'][:, rna_keep].copy()
    
    # ATAC
    atac_data = mdata['atac'].X.toarray() if hasattr(mdata['atac'].X, 'toarray') else mdata['atac'].X
    atac_std = np.std(atac_data, axis=0)
    atac_keep = atac_std > min_std
    n_atac_dropped = (~atac_keep).sum()
    
    if n_atac_dropped > 0:
        logger.info(f"    Dropping {n_atac_dropped} zero-variance ATAC features")
        mdata.mod['atac'] = mdata.mod['atac'][:, atac_keep].copy()
    
    logger.info(f"  Final dimensions: RNA={mdata['rna'].shape}, ATAC={mdata['atac'].shape}")
    return mdata

def run_mofa_integration(mdata, n_factors=20):
    """Run MOFA+ integration on RNA + ATAC"""
    logger.info(f"  Running MOFA+ integration ({n_factors} factors)...")
    
    try:
        from mofapy2.run.entry_point import entry_point
        
        rna_data = mdata['rna'].X.toarray() if hasattr(mdata['rna'].X, 'toarray') else mdata['rna'].X
        atac_data = mdata['atac'].X.toarray() if hasattr(mdata['atac'].X, 'toarray') else mdata['atac'].X
        
        logger.info(f"  RNA data: {rna_data.shape}")
        logger.info(f"  ATAC data: {atac_data.shape}")
        
        # Create nested list: [views][groups]
        data = [
            [rna_data],   # RNA view
            [atac_data]   # ATAC view
        ]
        
        ent = entry_point()
        
        ent.set_data_options(
            scale_groups=False,
            scale_views=True
        )
        
        ent.set_data_matrix(
            data,
            likelihoods=['gaussian', 'gaussian'],
            views_names=['RNA', 'ATAC'],
            groups_names=['group0'],
            samples_names=[mdata['rna'].obs_names.tolist()],
            features_names=[mdata['rna'].var_names.tolist(), mdata['atac'].var_names.tolist()]
        )
        
        ent.set_model_options(
            factors=n_factors,
            spikeslab_weights=True,
            ard_weights=True
        )
        
        ent.set_train_options(
            iter=1000,
            convergence_mode='fast',
            verbose=False,
            seed=42
        )
        
        ent.build()
        ent.run()
        
        expectations = ent.model.getExpectations()
        factors = expectations['Z']['E']
        
        logger.info(f"  ✅ MOFA+ complete: {factors.shape}")
        return factors
        
    except ImportError:
        logger.warning("  ⚠️  MOFA+ not available, using PCA on concatenated data")
        
        from sklearn.decomposition import PCA
        
        X_rna = mdata['rna'].X.toarray() if hasattr(mdata['rna'].X, 'toarray') else mdata['rna'].X
        X_atac = mdata['atac'].X.toarray() if hasattr(mdata['atac'].X, 'toarray') else mdata['atac'].X
        
        # Concatenate
        X_concat = np.hstack([X_rna, X_atac])
        
        # Z-score normalize
        X_concat = (X_concat - X_concat.mean(axis=0)) / (X_concat.std(axis=0) + 1e-8)
        
        # PCA
        pca = PCA(n_components=n_factors, random_state=42)
        factors = pca.fit_transform(X_concat)
        
        logger.info(f"  ✅ PCA fallback complete: {factors.shape}")
        return factors

def run_celltypist_annotation(adata):
    """Run CellTypist automated annotation"""
    logger.info("  Running CellTypist annotation...")
    
    try:
        import celltypist
        from celltypist import models
        
        logger.info("  Downloading CellTypist model...")
        models.download_models(force_update=False)
        
        logger.info("  Predicting cell types...")
        predictions = celltypist.annotate(
            adata,
            model='Immune_All_Low.pkl',
            majority_voting=True
        )
        
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
    logger.info("STEP 2.2: GROUND TRUTH LABELING (RNA + ATAC COMBINED)")
    logger.info("="*80)
    
    # Paths
    INPUT_PATH = 'Data/processed/PBMC/pbmc_filtered.h5mu'
    OUTPUT_PATH = 'Data/processed/PBMC/pbmc_mofa_annotated_RNA_ATAC.h5ad'
    MOFA_FACTORS_PATH = 'Results/Embeddings/RNA_ATAC/mofa_factors_RNA_ATAC.csv'
    
    # Parameters
    N_FACTORS = 20
    LEIDEN_RESOLUTION = 0.5
    
    try:
        # Load filtered data
        logger.info(f"\nLoading data from: {INPUT_PATH}")
        mdata = mu.read_h5mu(INPUT_PATH)
        logger.info(f"  RNA: {mdata['rna'].shape}")
        logger.info(f"  ATAC: {mdata['atac'].shape}")
        
        # Step 1: Save raw counts BEFORE normalization
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.1: Save Raw Counts")
        logger.info("-"*80)
        
        logger.info("  Saving raw counts for RNA and ATAC...")
        rna_raw_counts = mdata['rna'].X.copy()
        atac_raw_counts = mdata['atac'].X.copy()
        
        # Save original var_names for later indexing
        rna_original_var_names = mdata['rna'].var_names.copy()
        atac_original_var_names = mdata['atac'].var_names.copy()
        
        logger.info("  ✅ Raw counts saved")
        
        # Step 2: Normalize for annotation (use ALL genes for CellTypist!)
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.2: Normalize Data for Annotation")
        logger.info("-"*80)
        
        # CRITICAL FIX: Keep ALL genes for CellTypist annotation
        # Create a copy with all genes for annotation
        logger.info("  Creating full RNA data for CellTypist (ALL genes)...")
        adata_rna_full = mdata['rna'].copy()
        
        # Normalize RNA (ALL genes for CellTypist)
        sc.pp.normalize_total(adata_rna_full, target_sum=1e4)
        sc.pp.log1p(adata_rna_full)
        logger.info(f"  ✅ Normalized RNA with ALL {adata_rna_full.n_vars} genes for annotation")
        
        # Step 2b: Run CellTypist BEFORE feature selection
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.3: CellTypist Annotation (BEFORE feature selection)")
        logger.info("-"*80)
        
        # Run CellTypist on full RNA data
        adata_rna_full = run_celltypist_annotation(adata_rna_full)
        
        # Extract labels to transfer to filtered data later
        celltypist_labels = adata_rna_full.obs['celltypist_labels'].copy()
        celltypist_conf = adata_rna_full.obs.get('celltypist_conf', pd.Series(1.0, index=adata_rna_full.obs_names)).copy()
        
        # Now prepare data for MOFA+ with feature selection
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.4: Feature Selection for MOFA+ Integration")
        logger.info("-"*80)
        
        # Normalize and select features for MOFA+
        sc.pp.normalize_total(mdata['rna'], target_sum=1e4)
        sc.pp.log1p(mdata['rna'])
        sc.pp.highly_variable_genes(mdata['rna'], n_top_genes=2000)
        
        # Normalize ATAC
        sc.pp.normalize_total(mdata['atac'], target_sum=1e4)
        sc.pp.log1p(mdata['atac'])
        
        logger.info("  ✅ Normalization complete")
        
        # Select top features for MOFA+
        mdata = select_top_features(mdata, n_rna_features=2000, n_atac_features=5000)
        
        # Z-score normalize for MOFA+
        sc.pp.scale(mdata['rna'])
        sc.pp.scale(mdata['atac'], max_value=10)
        
        # Remove zero-variance features
        mdata = remove_zero_variance_features(mdata)
        
        # Step 3: MOFA+ Integration
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.5: MOFA+ Integration (RNA + ATAC)")
        logger.info("-"*80)
        
        mofa_factors = run_mofa_integration(mdata, n_factors=N_FACTORS)
        
        # Step 4: Create combined AnnData with RNA + ATAC features
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.6: Create Combined RNA+ATAC AnnData")
        logger.info("-"*80)
        
        # Get RNA and ATAC data (after normalization and filtering)
        rna_data = mdata['rna'].X.toarray() if hasattr(mdata['rna'].X, 'toarray') else mdata['rna'].X
        atac_data = mdata['atac'].X.toarray() if hasattr(mdata['atac'].X, 'toarray') else mdata['atac'].X
        
        # Get corresponding raw counts (subset to selected features using indices)
        # Find indices of selected features in original data
        rna_feature_indices = [i for i, name in enumerate(rna_original_var_names) if name in mdata['rna'].var_names]
        atac_feature_indices = [i for i, name in enumerate(atac_original_var_names) if name in mdata['atac'].var_names]
        
        # Subset raw counts using integer indices
        rna_raw_subset = rna_raw_counts[:, rna_feature_indices]
        atac_raw_subset = atac_raw_counts[:, atac_feature_indices]
        
        # Convert to dense if needed
        if hasattr(rna_raw_subset, 'toarray'):
            rna_raw_subset = rna_raw_subset.toarray()
        if hasattr(atac_raw_subset, 'toarray'):
            atac_raw_subset = atac_raw_subset.toarray()
        
        # Concatenate RNA + ATAC features
        X_combined = np.hstack([rna_data, atac_data])
        X_combined_raw = np.hstack([rna_raw_subset, atac_raw_subset])
        
        logger.info(f"  Combined normalized data: {X_combined.shape}")
        logger.info(f"  Combined raw counts: {X_combined_raw.shape}")
        
        # Create feature names
        rna_feature_names = [f"RNA_{name}" for name in mdata['rna'].var_names]
        atac_feature_names = [f"ATAC_{name}" for name in mdata['atac'].var_names]
        combined_feature_names = rna_feature_names + atac_feature_names
        
        # Create combined AnnData
        adata_combined = sc.AnnData(
            X=X_combined,
            obs=mdata['rna'].obs.copy(),
            var=pd.DataFrame(index=combined_feature_names)
        )
        
        # Add modality information to var
        adata_combined.var['modality'] = ['RNA'] * len(rna_feature_names) + ['ATAC'] * len(atac_feature_names)
        
        # Store raw counts in layers (critical for scVI!)
        adata_combined.layers['counts'] = X_combined_raw
        
        # Add MOFA factors
        adata_combined.obsm['X_mofa'] = mofa_factors
        
        logger.info("  ✅ Combined AnnData created")
        logger.info(f"     Features: {adata_combined.n_vars} ({len(rna_feature_names)} RNA + {len(atac_feature_names)} ATAC)")
        logger.info(f"     Raw counts in layers['counts']")
        
        # Save MOFA factors
        os.makedirs('Results/Embeddings', exist_ok=True)
        mofa_df = pd.DataFrame(
            mofa_factors,
            index=adata_combined.obs_names,
            columns=[f'Factor_{i+1}' for i in range(N_FACTORS)]
        )
        mofa_df.to_csv(MOFA_FACTORS_PATH)
        logger.info(f"  ✅ Saved MOFA factors: {MOFA_FACTORS_PATH}")
        
        # Step 5: Leiden Clustering
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.7: Leiden Clustering")
        logger.info("-"*80)
        
        sc.pp.neighbors(adata_combined, use_rep='X_mofa', n_neighbors=15, random_state=42)
        sc.tl.leiden(adata_combined, resolution=LEIDEN_RESOLUTION, random_state=42)
        sc.tl.umap(adata_combined, random_state=42)
        
        logger.info(f"  ✅ Clustering complete: {adata_combined.obs['leiden'].nunique()} clusters")
        
        # Step 6: Transfer CellTypist Annotations (already computed on ALL genes!)
        logger.info("\n" + "-"*80)
        logger.info("Step 2.2.8: Transfer CellTypist Annotations")
        logger.info("-"*80)
        
        # Transfer the labels we computed earlier on ALL genes
        logger.info("  Transferring labels from full gene annotation...")
        adata_combined.obs['celltypist_labels'] = celltypist_labels
        adata_combined.obs['celltypist_conf'] = celltypist_conf
        
        # Print summary
        logger.info("\n  Annotation Summary:")
        cell_type_counts = adata_combined.obs['celltypist_labels'].value_counts()
        for cell_type, count in cell_type_counts.head(10).items():
            logger.info(f"    {cell_type:30s}: {count:5d} cells")
        
        # Save annotation summary
        summary_df = pd.DataFrame({
            'cell_type': cell_type_counts.index,
            'count': cell_type_counts.values,
            'percentage': (cell_type_counts.values / adata_combined.n_obs * 100).round(2)
        })
        summary_path = 'Data/outputs/RNA_ATAC/ground_truth_summary_RNA_ATAC.csv'
        os.makedirs('Data/outputs/RNA_ATAC', exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n  ✅ Saved annotation summary: {summary_path}")
        
        # Save combined annotated data
        logger.info(f"\nSaving combined annotated data to: {OUTPUT_PATH}")
        os.makedirs('Data/processed/PBMC', exist_ok=True)
        adata_combined.write(OUTPUT_PATH)
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
    logger.info("STEP 2.2 (RNA + ATAC) COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Cells annotated: {adata_combined.n_obs}")
    logger.info(f"Features: {adata_combined.n_vars} (RNA + ATAC combined)")
    logger.info(f"  RNA features: {len(rna_feature_names)}")
    logger.info(f"  ATAC features: {len(atac_feature_names)}")
    logger.info(f"Cell types found: {adata_combined.obs['celltypist_labels'].nunique()}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"✅ Ready for scVI training with RNA+ATAC features!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
