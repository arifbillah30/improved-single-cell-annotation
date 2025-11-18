"""
FYDP Improved Method - Step 3.5: Supervised Classification with Factor Analysis (RNA-only)
===========================================================================================
Factor Analysis-based classification for RNA-only data.

Pipeline:
1. Load annotated data (RNA-only)
2. Bootstrap sampling (10 folds)
3. For each bootstrap:
   - Train/test split (80/20)
   - Factor Analysis embedding (35 components)
   - StandardScaler normalization
   - Apply SMOTE oversampling
   - Train classifiers (RF, SVM, XGBoost)
   - Save metrics

Input: Data/processed/AD/ad_mofa_annotated_RNA_only.h5ad
Output: Results/Metrics/AD_FA_RNA_ONLY/classification_results_FA_*.csv
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
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# Setup logging
log_file = 'Logs/step3_5_FA_RNA_ONLY.log'
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
N_COMPONENTS = 35  # RNA only
TEST_SIZE = 0.2
RANDOM_SEED = 42
EMBEDDING_NAME = 'FA'

# Output directories
RESULTS_DIR = 'Results/Metrics/AD/FA_RNA_ONLY'
MODELS_DIR = 'Models/AD/FA_RNA_ONLY'

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

def compute_fa_embedding(X, n_components=35):
    """
    Compute Factor Analysis embedding.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of FA components
    
    Returns:
        X_fa: FA embedding (n_samples, n_components)
        fa_model: Fitted FactorAnalysis model
    """
    logger.info(f"    Running Factor Analysis ({X.shape[1]} features → {n_components} components)...")
    fa_model = FactorAnalysis(n_components=n_components, random_state=RANDOM_SEED)
    X_fa = fa_model.fit_transform(X)
    logger.info(f"    ✅ FA embedding: {X_fa.shape}")
    return X_fa, fa_model

def apply_smote(X, y):
    """Apply SMOTE oversampling (FYDP enhancement)"""
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
    logger.info("STEP 3.5: SUPERVISED CLASSIFICATION - FACTOR ANALYSIS (RNA-only)")
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
    logger.info(f"  Features: {adata.n_vars}")
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
    
    # Get feature matrix
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED + bootstrap_idx,
            stratify=y_encoded
        )
        
        logger.info(f"  Train: {X_train.shape[0]} cells")
        logger.info(f"  Test:  {X_test.shape[0]} cells")
        
        # Compute Factor Analysis embedding
        logger.info(f"  Computing {EMBEDDING_NAME} ({N_COMPONENTS} components)...")
        
        try:
            # Train FA on training data
            X_train_emb, fa_model = compute_fa_embedding(X_train, n_components=N_COMPONENTS)
            
            # Transform test data
            logger.info("    Transforming test set with trained FA model...")
            X_test_emb = fa_model.transform(X_test)
            logger.info(f"    ✅ Test embedding: {X_test_emb.shape}")
            
            # Apply StandardScaler
            logger.info("    Applying StandardScaler to embeddings...")
            scaler = StandardScaler()
            X_train_emb = scaler.fit_transform(X_train_emb)
            X_test_emb = scaler.transform(X_test_emb)
            logger.info(f"    ✅ Scaled embeddings: train {X_train_emb.shape}, test {X_test_emb.shape}")
            
        except Exception as e:
            logger.error(f"❌ Factor Analysis computation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        # Apply SMOTE (FYDP enhancement)
        logger.info("  Applying SMOTE (FYDP enhancement)...")
        try:
            X_train_smote, y_train_smote = apply_smote(X_train_emb, y_train)
            logger.info(f"    Before SMOTE: {X_train_emb.shape[0]} samples")
            logger.info(f"    After SMOTE:  {X_train_smote.shape[0]} samples")
        except Exception as e:
            logger.warning(f"    ⚠️  SMOTE failed: {e}. Using original data.")
            X_train_smote = X_train_emb
            y_train_smote = y_train
        
        # Train each classifier
        for clf_name, clf in CLASSIFIERS.items():
            logger.info(f"\n  Classifier: {clf_name}")
            
            clf_start = datetime.now()
            
            try:
                # Train
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
                    
            except Exception as e:
                logger.error(f"    ❌ Classification failed: {e}")
                continue
    
    # Save results
    if not all_results:
        logger.error("❌ No results to save. All bootstraps failed.")
        return
    
    results_df = pd.DataFrame(all_results)
    results_path = f"{RESULTS_DIR}/classification_results_{EMBEDDING_NAME}_detailed.csv"
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
    summary_path = f"{RESULTS_DIR}/classification_results_{EMBEDDING_NAME}_summary.csv"
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
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info(f"STEP 3.5 ({EMBEDDING_NAME} RNA-only) COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Total models trained: {len(all_results)}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
