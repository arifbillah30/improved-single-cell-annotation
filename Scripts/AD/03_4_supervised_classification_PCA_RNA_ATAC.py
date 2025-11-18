"""
FYDP Improved Method - Step 3.4: Supervised Classification with PCA (RNA+ATAC)
================================================================================
Fast PCA-based classification for RNA+ATAC combined data.

Pipeline:
1. Load annotated data (RNA+ATAC combined)
2. Bootstrap sampling (10 folds)
3. For each bootstrap:
   - Train/test split (80/20)
   - PCA embedding (35 components per modality = 70 total)
   - Apply SMOTE oversampling
   - Train classifiers (RF, SVM, XGBoost)
   - Save metrics

Input: Data/processed/AD/ad_mofa_annotated_RNA_ATAC.h5ad
Output: Results/Metrics/AD_PCA_RNA_ATAC/classification_results_PCA_*.csv
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
log_file = 'Logs/step3_4_PCA_RNA_ATAC.log'
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
N_COMPONENTS = 35  # Per modality (70 total)
TEST_SIZE = 0.2
RANDOM_SEED = 42
EMBEDDING_NAME = 'PCA'

# Output directories
RESULTS_DIR = 'Results/Metrics/AD/PCA_RNA_ATAC'
MODELS_DIR = 'Models/AD/PCA_RNA_ATAC'

# Classifiers
CLASSIFIERS = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
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

def compute_pca_embedding_rna_atac(adata, n_components=35):
    """
    Compute PCA embeddings separately for RNA and ATAC, then concatenate.
    
    Args:
        adata: AnnData with var['modality'] indicating RNA vs ATAC
        n_components: Number of PCA components per modality
    
    Returns:
        concatenated_embeddings: (n_cells, n_components*2)
    """
    logger.info(f"    Computing PCA embeddings separately for RNA and ATAC ({n_components} components each)...")
    
    # Split by modality
    rna_mask = adata.var['modality'] == 'RNA'
    atac_mask = adata.var['modality'] == 'ATAC'
    
    logger.info(f"      RNA features: {rna_mask.sum()}")
    logger.info(f"      ATAC features: {atac_mask.sum()}")
    
    # Get normalized data (use .X which should be normalized)
    X_rna = adata[:, rna_mask].X
    X_atac = adata[:, atac_mask].X
    
    # Convert sparse to dense if needed
    if hasattr(X_rna, 'toarray'):
        X_rna = X_rna.toarray()
    if hasattr(X_atac, 'toarray'):
        X_atac = X_atac.toarray()
    
    # PCA on RNA
    logger.info(f"      Running PCA on RNA ({X_rna.shape[1]} features → {n_components} components)...")
    pca_rna = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_rna_pca = pca_rna.fit_transform(X_rna)
    logger.info(f"      ✅ RNA PCA: {X_rna_pca.shape}, explained variance: {pca_rna.explained_variance_ratio_.sum():.3f}")
    
    # PCA on ATAC
    logger.info(f"      Running PCA on ATAC ({X_atac.shape[1]} features → {n_components} components)...")
    pca_atac = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_atac_pca = pca_atac.fit_transform(X_atac)
    logger.info(f"      ✅ ATAC PCA: {X_atac_pca.shape}, explained variance: {pca_atac.explained_variance_ratio_.sum():.3f}")
    
    # Concatenate
    X_combined = np.hstack([X_rna_pca, X_atac_pca])
    logger.info(f"    ✅ Combined PCA embedding: {X_combined.shape} (RNA + ATAC)")
    
    return X_combined, pca_rna, pca_atac

def apply_smote(X, y):
    """Apply SMOTE oversampling"""
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def main():
    # Change to the FYDP_Improved_Method directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Scripts/AD/
    scripts_dir = os.path.dirname(script_dir)  # Scripts/
    project_dir = os.path.dirname(scripts_dir)  # FYDP_Improved_Method/
    os.chdir(project_dir)
    
    # Create output directories after changing to project directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    start_time = datetime.now()
    
    logger.info("\n" + "="*80)
    logger.info("STEP 3.4: SUPERVISED CLASSIFICATION - PCA EMBEDDING")
    logger.info("="*80)
    logger.info("")
    
    # Load data
    data_path = 'Data/processed/AD/ad_mofa_annotated_RNA_ATAC.h5ad'
    logger.info(f"Loading annotated data from: {data_path}")
    
    try:
        adata = sc.read_h5ad(data_path)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {data_path}")
        logger.error("Please run Step 2.2 first to generate annotated data.")
        return
    
    logger.info(f"  Original cells: {adata.n_obs:,}")
    logger.info(f"  Genes: {adata.n_vars}")
    logger.info(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    
    # REDUCE DATASET SIZE for faster training
    MAX_CELLS = 5000
    if adata.n_obs > MAX_CELLS:
        logger.info(f"\n⚡ Reducing dataset to {MAX_CELLS:,} cells for faster training...")
        logger.info(f"   Using stratified sampling to maintain class proportions")
        
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        subset_indices, _ = train_test_split(
            np.arange(adata.n_obs),
            train_size=MAX_CELLS,
            random_state=RANDOM_SEED,
            stratify=adata.obs['cell_type']
        )
        adata = adata[subset_indices].copy()
        logger.info(f"   ✅ Reduced to {adata.n_obs:,} cells")
    
    y = adata.obs['cell_type'].values
    
    # Encode labels
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
        
        # Train/test split
        train_indices, test_indices = train_test_split(
            np.arange(adata.n_obs),
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED + bootstrap_idx,
            stratify=y_encoded
        )
        
        adata_train = adata[train_indices].copy()
        adata_test = adata[test_indices].copy()
        
        y_train = y_encoded[train_indices]
        y_test = y_encoded[test_indices]
        
        logger.info(f"  Train: {adata_train.n_obs} cells")
        logger.info(f"  Test:  {adata_test.n_obs} cells")
        
        # Compute PCA embedding
        logger.info(f"  Computing {EMBEDDING_NAME} ({N_COMPONENTS} components for RNA + {N_COMPONENTS} for ATAC)...")
        
        try:
            # Train PCA on training data
            X_train_emb, pca_rna, pca_atac = compute_pca_embedding_rna_atac(adata_train, n_components=N_COMPONENTS)
            
            # Transform test data with trained PCAs
            logger.info("    Transforming test set with trained PCA models...")
            rna_mask = adata_test.var['modality'] == 'RNA'
            atac_mask = adata_test.var['modality'] == 'ATAC'
            
            X_test_rna = adata_test[:, rna_mask].X
            X_test_atac = adata_test[:, atac_mask].X
            
            if hasattr(X_test_rna, 'toarray'):
                X_test_rna = X_test_rna.toarray()
            if hasattr(X_test_atac, 'toarray'):
                X_test_atac = X_test_atac.toarray()
            
            X_test_rna_pca = pca_rna.transform(X_test_rna)
            X_test_atac_pca = pca_atac.transform(X_test_atac)
            X_test_emb = np.hstack([X_test_rna_pca, X_test_atac_pca])
            logger.info(f"    ✅ Test embedding: {X_test_emb.shape}")
            
            # Apply StandardScaler
            logger.info("    Applying StandardScaler to embeddings...")
            scaler = StandardScaler()
            X_train_emb = scaler.fit_transform(X_train_emb)
            X_test_emb = scaler.transform(X_test_emb)
            logger.info(f"    ✅ Scaled embeddings: train {X_train_emb.shape}, test {X_test_emb.shape}")
            
        except Exception as e:
            logger.error(f"❌ PCA computation failed: {e}")
            logger.error("Cannot proceed with PCA embedding")
            return
        
        # Apply SMOTE - FYDP Enhancement
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
    df_results = pd.DataFrame(all_results)
    
    # Detailed results
    results_file = f'{RESULTS_DIR}/classification_results_PCA_detailed.csv'
    df_results.to_csv(results_file, index=False)
    logger.info(f"\n✅ Detailed results saved: {results_file}")
    
    # Summary statistics
    summary = df_results.groupby(['Embedding', 'Classifier']).agg({
        'Accuracy': ['mean', 'std'],
        'F1_Macro': ['mean', 'std'],
        'Precision_Macro': ['mean', 'std'],
        'Recall_Macro': ['mean', 'std'],
        'Training_Time': ['mean', 'sum']
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    
    summary_file = f'{RESULTS_DIR}/classification_results_PCA_summary.csv'
    summary.to_csv(summary_file, index=False)
    logger.info(f"✅ Summary results saved: {summary_file}")
    
    # Print best results
    logger.info("\n" + "="*80)
    logger.info("BEST RESULTS")
    logger.info("="*80)
    
    best_idx = summary['F1_Macro_mean'].idxmax()
    best_row = summary.iloc[best_idx]
    
    logger.info(f"\nBest Model: {best_row['Classifier']}")
    logger.info(f"  F1 Score: {best_row['F1_Macro_mean']:.4f} ± {best_row['F1_Macro_std']:.4f}")
    logger.info(f"  Accuracy: {best_row['Accuracy_mean']:.4f} ± {best_row['Accuracy_std']:.4f}")
    
    # Completion
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*80)
    logger.info("STEP 3.4 (PCA) COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Total models trained: {len(all_results)}")
    logger.info(f"Results: {results_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

if __name__ == '__main__':
    main()
