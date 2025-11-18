"""
FYDP Improved Method - Step 3.4: Supervised Classification with PCA (RNA-only)
================================================================================
Fast PCA-based classification for RNA-only data.

Pipeline:
1. Load annotated data (RNA-only)
2. Bootstrap sampling (10 folds)
3. For each bootstrap:
   - Train/test split (80/20)
   - PCA embedding (35 components)
   - Apply SMOTE oversampling
   - Train classifiers (RF, SVM, XGBoost)
   - Save metrics

Input: Data/processed/AD/ad_mofa_annotated_RNA_only.h5ad
Output: Results/Metrics/AD_PCA_RNA_only/classification_results_PCA_*.csv
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
log_file = 'Logs/step3_4_PCA_RNA_only.log'
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
EMBEDDING_NAME = 'PCA'

# Output directories
RESULTS_DIR = 'Results/Metrics/AD/PCA_RNA_ONLY'
MODELS_DIR = 'Models/AD/PCA_RNA_ONLY'

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

def compute_pca_embedding(X, n_components=35):
    """
    Compute PCA embedding.
    
    Args:
        X: Data matrix
        n_components: Number of PCA components
    
    Returns:
        embeddings: (n_cells, n_components)
        pca: Fitted PCA object
    """
    logger.info(f"    Computing PCA embedding ({n_components} components)...")
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # PCA
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    logger.info(f"    ✅ PCA: {X_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_pca, pca

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
    logger.info("STEP 3.4: SUPERVISED CLASSIFICATION - PCA EMBEDDING (RNA-only)")
    logger.info("="*80)
    logger.info("")
    
    # Load data
    data_path = 'Data/processed/AD/ad_mofa_annotated_RNA_only.h5ad'
    logger.info(f"Loading annotated data from: {data_path}")
    
    try:
        adata = sc.read_h5ad(data_path)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {data_path}")
        logger.error("Please run Step 2.1 first to generate annotated data.")
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
    
    # Get normalized data from .X (should be normalized and scaled)
    X = adata.X
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
    logger.info("")
    
    for bootstrap_idx in range(N_BOOTSTRAPS):
        logger.info("="*80)
        logger.info(f"Bootstrap {bootstrap_idx+1}/{N_BOOTSTRAPS}")
        logger.info("="*80)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED + bootstrap_idx,
            stratify=y_encoded
        )
        
        logger.info(f"  Train: {X_train.shape[0]} cells")
        logger.info(f"  Test:  {X_test.shape[0]} cells")
        
        # Compute PCA embedding
        logger.info(f"  Computing PCA ({N_COMPONENTS} components)...")
        X_train_pca, pca = compute_pca_embedding(X_train, n_components=N_COMPONENTS)
        
        # Transform test set
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        X_test_pca = pca.transform(X_test)
        logger.info(f"    ✅ Test embedding: {X_test_pca.shape}")
        
        # Apply StandardScaler
        logger.info(f"    Applying StandardScaler to embeddings...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)
        logger.info(f"    ✅ Scaled embeddings: train {X_train_scaled.shape}, test {X_test_scaled.shape}")
        
        # Apply SMOTE
        logger.info("  Applying SMOTE...")
        X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)
        logger.info(f"    Before SMOTE: {X_train_scaled.shape[0]} samples")
        logger.info(f"    After SMOTE:  {X_train_smote.shape[0]} samples")
        
        # Train and evaluate classifiers
        for clf_name, clf in CLASSIFIERS.items():
            logger.info(f"\n  Classifier: {clf_name}")
            
            # Train
            start_clf = datetime.now()
            clf.fit(X_train_smote, y_train_smote)
            train_time = (datetime.now() - start_clf).total_seconds()
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            precision_macro = precision_score(y_test, y_pred, average='macro')
            recall_macro = recall_score(y_test, y_pred, average='macro')
            
            logger.info(f"    Accuracy: {accuracy:.4f}")
            logger.info(f"    F1 (macro): {f1_macro:.4f}")
            logger.info(f"    Precision (macro): {precision_macro:.4f}")
            logger.info(f"    Recall (macro): {recall_macro:.4f}")
            logger.info(f"    Time: {train_time:.2f}s")
            
            # Save results
            all_results.append({
                'bootstrap': bootstrap_idx + 1,
                'classifier': clf_name,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'training_time': train_time,
                'n_train': X_train_smote.shape[0],
                'n_test': X_test_scaled.shape[0]
            })
            
            # Save model
            model_path = f"{MODELS_DIR}/{clf_name}_bootstrap_{bootstrap_idx+1}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)
        
        logger.info("")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    detailed_path = f"{RESULTS_DIR}/classification_results_{EMBEDDING_NAME}_detailed.csv"
    results_df.to_csv(detailed_path, index=False)
    logger.info(f"✅ Detailed results saved: {detailed_path}")
    
    # Compute and save summary statistics
    summary_stats = results_df.groupby('classifier').agg({
        'accuracy': ['mean', 'std'],
        'f1_macro': ['mean', 'std'],
        'precision_macro': ['mean', 'std'],
        'recall_macro': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).round(4)
    
    summary_path = f"{RESULTS_DIR}/classification_results_{EMBEDDING_NAME}_summary.csv"
    summary_stats.to_csv(summary_path)
    logger.info(f"✅ Summary results saved: {summary_path}")
    
    # Print best results
    logger.info("\n" + "="*80)
    logger.info("BEST RESULTS")
    logger.info("="*80)
    logger.info("")
    
    best_f1_idx = results_df.groupby('classifier')['f1_macro'].mean().idxmax()
    best_f1 = results_df[results_df['classifier'] == best_f1_idx]['f1_macro'].mean()
    best_f1_std = results_df[results_df['classifier'] == best_f1_idx]['f1_macro'].std()
    best_acc = results_df[results_df['classifier'] == best_f1_idx]['accuracy'].mean()
    best_acc_std = results_df[results_df['classifier'] == best_f1_idx]['accuracy'].std()
    
    logger.info(f"Best Model: {best_f1_idx}")
    logger.info(f"  F1 Score: {best_f1:.4f} ± {best_f1_std:.4f}")
    logger.info(f"  Accuracy: {best_acc:.4f} ± {best_acc_std:.4f}")
    
    # Duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info(f"STEP 3.4 (PCA RNA-only) COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Total models trained: {len(all_results)}")
    logger.info(f"Results: {detailed_path}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

if __name__ == '__main__':
    main()
