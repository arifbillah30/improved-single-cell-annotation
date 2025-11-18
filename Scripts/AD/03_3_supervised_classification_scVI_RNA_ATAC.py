"""
FYDP Improved Method - Step 3.3: Supervised Classification with scVI (RNA+ATAC)
=================================================================================
Run bootstrap classification with SMOTE oversampling using scVI embedding.

Pipeline:
1. Load annotated data (RNA+ATAC combined)
2. Bootstrap sampling (10 folds)
3. For each bootstrap:
   - Train/test split (80/20)
   - Compute scVI embedding on RAW counts
   - Apply SMOTE oversampling
   - Train classifiers (RF, SVM, XGBoost)
   - Save metrics

Note: scVI requires raw counts in adata.layers['counts']
If this is not available, the script will fall back to PCA.

Input: Data/processed/AD/ad_mofa_annotated_RNA_ATAC.h5ad
Output: Results/Metrics/AD_scVI_RNA_ATAC/classification_results_scVI_*.csv
"""

import scanpy as sc
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# Setup logging
log_file = 'Logs/step3_3_scVI_RNA_ATAC.log'
os.makedirs('Logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
N_BOOTSTRAPS = 10
N_COMPONENTS = 35
TEST_SIZE = 0.2
RANDOM_SEED = 42
EMBEDDING_NAME = 'scVI'

# Classifiers
CLASSIFIERS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',  # Reference uses rbf, not poly
        class_weight='balanced',
        probability=True,
        random_state=RANDOM_SEED
    ),
    'XGBoost': XGBClassifier(
        objective='multi:softprob',
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        tree_method='hist',
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
}

def compute_scvi_embedding_rna_atac(adata, n_latent=35):
    """
    Compute scVI embeddings SEPARATELY for RNA and ATAC, then concatenate.
    This matches the reference paper approach in Utils.py scvi_process()
    
    adata should have:
    - var['modality'] column indicating 'RNA' or 'ATAC'
    - layers['counts'] with raw counts for all features
    
    Returns: (concatenated_latent_representation, rna_model, atac_model)
    """
    logger.info(f"    Computing scVI embeddings separately for RNA and ATAC ({n_latent} components each)...")
    
    try:
        import scvi
        import torch
        
        # Check if adata has layers["counts"] with raw counts
        if 'counts' not in adata.layers:
            raise ValueError("adata.layers['counts'] not found - scVI requires raw counts")
        
        if 'modality' not in adata.var.columns:
            raise ValueError("adata.var['modality'] not found - need to identify RNA vs ATAC features")
        
        # Split into RNA and ATAC
        rna_mask = adata.var['modality'] == 'RNA'
        atac_mask = adata.var['modality'] == 'ATAC'
        
        logger.info(f"      RNA features: {rna_mask.sum()}")
        logger.info(f"      ATAC features: {atac_mask.sum()}")
        
        # Create separate AnnData objects for RNA and ATAC
        adata_rna = adata[:, rna_mask].copy()
        adata_atac = adata[:, atac_mask].copy()
        
        # Train scVI on RNA (100 epochs to match reference paper)
        # CRITICAL: early_stopping_patience=45 causes it to train for 400 epochs!
        # We explicitly disable early_stopping to force exactly 100 epochs.
        logger.info(f"      Training scVI on RNA ({adata_rna.n_vars} features)...")
        scvi.model.SCVI.setup_anndata(adata_rna, layer="counts")
        model_rna = scvi.model.SCVI(adata_rna, n_latent=n_latent)
        model_rna.train(
            max_epochs=100,
            early_stopping=False,  # CRITICAL: Disable early stopping
            check_val_every_n_epoch=None,
        )
        latent_rna = model_rna.get_latent_representation()
        logger.info(f"      ✅ RNA scVI embedding: {latent_rna.shape}")
        
        # Train scVI on ATAC (100 epochs to match reference paper)
        logger.info(f"      Training scVI on ATAC ({adata_atac.n_vars} features)...")
        scvi.model.SCVI.setup_anndata(adata_atac, layer="counts")
        model_atac = scvi.model.SCVI(adata_atac, n_latent=n_latent)
        model_atac.train(
            max_epochs=100,
            early_stopping=False,  # CRITICAL: Disable early stopping
            check_val_every_n_epoch=None,
        )
        latent_atac = model_atac.get_latent_representation()
        logger.info(f"      ✅ ATAC scVI embedding: {latent_atac.shape}")
        
        # Concatenate RNA and ATAC embeddings
        latent_combined = np.hstack([latent_rna, latent_atac])
        
        # Check for NaN values
        if np.isnan(latent_combined).any():
            raise ValueError("scVI produced NaN values in latent representation")
        
        logger.info(f"    ✅ Combined scVI embedding: {latent_combined.shape} (RNA + ATAC)")
        return latent_combined, model_rna, model_atac
        
    except ImportError:
        logger.warning("    ⚠️  scVI not available, using PCA fallback")
        raise ImportError("scVI package not found")
        
    except Exception as e:
        logger.warning(f"    ⚠️  scVI failed: {e}. Cannot proceed.")
        raise

def apply_smote(X_train, y_train, k_neighbors=5):
    """Apply SMOTE oversampling"""
    min_class_size = pd.Series(y_train).value_counts().min()
    
    if min_class_size < k_neighbors + 1:
        k_neighbors = max(1, min_class_size - 1)
        logger.info(f"      Adjusting SMOTE k_neighbors to {k_neighbors} (min class size: {min_class_size})")
    
    try:
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    except ValueError as e:
        logger.warning(f"      ⚠️  SMOTE failed: {e}. Using original data.")
        return X_train, y_train

def main():
    start_time = datetime.now()
    logger.info("="*80)
    logger.info(f"FYDP IMPROVED METHOD - STEP 3.3: SUPERVISED CLASSIFICATION (scVI RNA+ATAC)")
    logger.info("="*80)
    logger.info(f"AD Neuronal Cells Dataset")
    logger.info(f"Embedding: {EMBEDDING_NAME}, Components: {N_COMPONENTS}, Bootstrap: {N_BOOTSTRAPS}")
    logger.info("")
    
    # Paths
    INPUT_PATH = 'Data/processed/AD/ad_mofa_annotated_RNA_ATAC.h5ad'
    OUTPUT_DIR = 'Results/Metrics/AD/scVI_RNA_ATAC'
    MODELS_DIR = 'Models/AD/scVI_RNA_ATAC'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        # Load annotated data
        logger.info(f"\nLoading annotated data from: {INPUT_PATH}")
        adata = sc.read_h5ad(INPUT_PATH)
        logger.info(f"  Original cells: {adata.n_obs:,}")
        logger.info(f"  Original features: {adata.n_vars:,}")
        logger.info(f"  Cell types: {adata.obs['cell_type'].nunique()}")
        
        # REDUCE DATASET SIZE for faster training (50% of previous)
        MAX_CELLS = 5000
        if adata.n_obs > MAX_CELLS:
            logger.info(f"\n⚡ Reducing dataset to {MAX_CELLS:,} cells for faster training...")
            logger.info(f"   (Original: {adata.n_obs:,} cells)")
            
            # Stratified sampling to keep class proportions
            np.random.seed(RANDOM_SEED)
            from sklearn.model_selection import train_test_split
            
            # Get stratified subset
            subset_indices, _ = train_test_split(
                np.arange(adata.n_obs),
                train_size=MAX_CELLS,
                random_state=RANDOM_SEED,
                stratify=adata.obs['cell_type']
            )
            
            adata = adata[subset_indices].copy()
            logger.info(f"   ✅ Reduced to {adata.n_obs:,} cells (stratified sampling)")
        
        # Check for raw counts in layers
        if 'counts' not in adata.layers:
            logger.error("❌ adata.layers['counts'] not found!")
            logger.error("scVI requires raw counts stored before normalization.")
            logger.error("Please re-run Step 2 with updated code that saves raw counts.")
            return
        
        logger.info("  ✅ Found raw counts in adata.layers['counts']")
        
        y = adata.obs['cell_type'].values
        
        # Encode labels for XGBoost
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info("\nClass distribution:")
        for cell_type, count in pd.Series(y).value_counts().head(10).items():
            logger.info(f"  {cell_type:30s}: {count:5d}")
        
        # Store all results
        all_results = []
        
        # Bootstrap loop
        logger.info(f"\n" + "="*80)
        logger.info(f"BOOTSTRAP CLASSIFICATION ({N_BOOTSTRAPS} folds)")
        logger.info("="*80)
        
        for bootstrap_idx in range(N_BOOTSTRAPS):
            logger.info(f"\n{'='*80}")
            logger.info(f"Bootstrap {bootstrap_idx + 1}/{N_BOOTSTRAPS}")
            logger.info(f"{'='*80}")
            
            # Train/test split using indices to properly subset layers
            train_indices, test_indices = train_test_split(
                np.arange(adata.n_obs),
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED + bootstrap_idx,
                stratify=y_encoded
            )
            
            # Create train/test AnnData subsets
            adata_train = adata[train_indices].copy()
            adata_test = adata[test_indices].copy()
            
            y_train = y_encoded[train_indices]
            y_test = y_encoded[test_indices]
            
            logger.info(f"  Train: {adata_train.n_obs} cells")
            logger.info(f"  Test:  {adata_test.n_obs} cells")
            
            # Compute scVI embedding on training data (separately for RNA and ATAC)
            logger.info(f"  Computing {EMBEDDING_NAME} ({N_COMPONENTS} components for RNA + {N_COMPONENTS} for ATAC)...")
            try:
                # Train scVI separately on RNA and ATAC, then concatenate embeddings
                X_train_emb, model_rna, model_atac = compute_scvi_embedding_rna_atac(adata_train, n_latent=N_COMPONENTS)
                
                # Use the SAME trained models to transform test data
                logger.info("    Transforming test set with trained scVI models...")
                
                # Split test data
                rna_mask = adata_test.var['modality'] == 'RNA'
                atac_mask = adata_test.var['modality'] == 'ATAC'
                
                adata_test_rna = adata_test[:, rna_mask].copy()
                adata_test_atac = adata_test[:, atac_mask].copy()
                
                # Transform with trained models
                import scvi
                X_test_rna = model_rna.get_latent_representation(adata_test_rna)
                X_test_atac = model_atac.get_latent_representation(adata_test_atac)
                X_test_emb = np.hstack([X_test_rna, X_test_atac])
                logger.info(f"    ✅ Test embedding: {X_test_emb.shape}")
                
                # Apply StandardScaler (CRITICAL - matches reference paper Utils.py line 577-579)
                logger.info("    Applying StandardScaler to embeddings...")
                scaler = StandardScaler()
                X_train_emb = scaler.fit_transform(X_train_emb)
                X_test_emb = scaler.transform(X_test_emb)
                logger.info(f"    ✅ Scaled embeddings: train {X_train_emb.shape}, test {X_test_emb.shape}")
                
            except Exception as e:
                logger.error(f"❌ scVI computation failed: {e}")
                logger.error("Cannot proceed with scVI embedding")
                return
            
            # Apply SMOTE - FYDP Enhancement over reference paper
            # Reference paper doesn't use SMOTE, but we test if it improves performance
            logger.info(f"  Applying SMOTE (FYDP enhancement)...")
            X_train_smote, y_train_smote = apply_smote(X_train_emb, y_train)
            logger.info(f"    Before SMOTE: {X_train_emb.shape[0]} samples")
            logger.info(f"    After SMOTE:  {X_train_smote.shape[0]} samples")
            
            # Train each classifier
            for clf_name, clf in CLASSIFIERS.items():
                logger.info(f"\n  Classifier: {clf_name}")
                
                clf_start = datetime.now()
                
                # Train with SMOTE-enhanced data
                clf.fit(X_train_smote, y_train_smote)
                
                # Predict
                y_pred = clf.predict(X_test_emb)
                
                clf_duration = (datetime.now() - clf_start).total_seconds()
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                
                logger.info(f"    Accuracy: {accuracy:.4f}")
                logger.info(f"    F1 (macro): {f1_macro:.4f}")
                logger.info(f"    Precision (macro): {precision_macro:.4f}")
                logger.info(f"    Recall (macro): {recall_macro:.4f}")
                logger.info(f"    Time: {clf_duration:.2f}s")
                
                # Store results
                all_results.append({
                    'Bootstrap': bootstrap_idx,
                    'Embedding': EMBEDDING_NAME,
                    'Classifier': clf_name,
                    'Accuracy': accuracy,
                    'F1_Macro': f1_macro,
                    'Precision_Macro': precision_macro,
                    'Recall_Macro': recall_macro,
                    'Training_Time': clf_duration,
                    'Train_Samples_Before_SMOTE': len(y_train),
                    'Train_Samples_After_SMOTE': len(y_train_smote),
                    'Test_Samples': len(y_test)
                })
                
                # Save model
                model_path = f"{MODELS_DIR}/{clf_name}_bootstrap{bootstrap_idx}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(clf, f)
        
        # Save all results
        results_df = pd.DataFrame(all_results)
        results_path = f"{OUTPUT_DIR}/classification_results_{EMBEDDING_NAME}_detailed.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n✅ Detailed results saved: {results_path}")
        
        # Aggregate results
        summary = results_df.groupby(['Embedding', 'Classifier']).agg({
            'Accuracy': ['mean', 'std'],
            'F1_Macro': ['mean', 'std'],
            'Precision_Macro': ['mean', 'std'],
            'Recall_Macro': ['mean', 'std'],
            'Training_Time': ['mean', 'sum']
        }).reset_index()
        
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        summary_path = f"{OUTPUT_DIR}/classification_results_{EMBEDDING_NAME}_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"✅ Summary results saved: {summary_path}")
        
        # Print best results
        logger.info("\n" + "="*80)
        logger.info("BEST RESULTS")
        logger.info("="*80)
        
        best_idx = summary['F1_Macro_mean'].idxmax()
        best = summary.loc[best_idx]
        
        logger.info(f"\nBest Model: {best['Classifier']}")
        logger.info(f"  F1 Score: {best['F1_Macro_mean']:.4f} ± {best['F1_Macro_std']:.4f}")
        logger.info(f"  Accuracy: {best['Accuracy_mean']:.4f} ± {best['Accuracy_std']:.4f}")
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.info("\nPlease run Step 2 first:")
        logger.info("  python Scripts/02_ground_truth_labeling.py")
        return
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info(f"STEP 3.3 ({EMBEDDING_NAME}) COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Total models trained: {len(all_results)}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
