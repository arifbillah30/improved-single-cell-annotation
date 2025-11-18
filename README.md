# Improved Single-Cell ATAC+RNA Annotation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Improved single-cell ATAC+RNA classification achieving **99.96% F1** for neuronal cells using SMOTE oversampling, automated CellTypist annotation, and proper scVI configuration. Handles severe class imbalance (180:1 ratio) with comprehensive evaluation across PCA/scVI/Factor Analysis methods.

---

## Key Results

| Dataset | Best Method | F1 Score | Improvement |
|---------|-------------|----------|-------------|
| **AD Neuronal** | PCA-RandomForest | **99.96%** | **+14.7%** over baseline |
| **PBMC Immune** | FA-XGBoost | **93.13%** | Comparable to state-of-art |

**Challenge Solved:** Reference paper achieved only 87.1% F1 for neuronal cells. We achieved **near-perfect classification** with the same data.

---

## Quick Start

### Installation

\`\`\`bash
# Clone repository
git clone https://github.com/arifbillah30/improved-single-cell-annotation.git
cd improved-single-cell-annotation

# Create conda environment
conda create -n multiome python=3.10
conda activate multiome

# Install dependencies
pip install scanpy muon scvi-tools celltypist
pip install scikit-learn xgboost imbalanced-learn
pip install pandas numpy matplotlib seaborn
\`\`\`

### Run Pipeline

\`\`\`bash
# Stage 1: Data Preparation (~2 min)
python Scripts/01_data_preparation.py

# Stage 2: Ground Truth Annotation (~5 min)
python Scripts/02_2_ground_truth_labeling_RNA_ATAC.py

# Stage 3: Classification (~1 hour for all methods)
python Scripts/03_3_supervised_classification_scVI_RNA_ATAC.py
python Scripts/03_4_supervised_classification_PCA_RNA_ATAC.py
python Scripts/03_5_supervised_classification_FA_RNA_ATAC.py

# Generate visualizations
python Scripts/pbmc_all_figures.py
\`\`\`

---

## Methodology

### Three-Pronged Innovation

**1. Automated Annotation (CellTypist)**
- Uses ALL 26,000 genes (not subset)
- 69%+ gene matching (vs 39.7% before fix)
- Removes manual annotation bias

**2. Dimensionality Reduction**
- PCA: 50 components
- scVI: 35 RNA + 35 ATAC = 70 dims
- Factor Analysis: 50 components

**3. SMOTE in Embedding Space (KEY INNOVATION!)**
- Balances 180:1 ratio to 1:1
- 4,000 to 13,800 balanced samples
- Applied in low-dim space (35-70 dims)

**4. Classification**
- Random Forest (n=100 trees)
- SVM (RBF kernel)
- XGBoost (depth=6)
- 10-fold bootstrap validation

### Why This Works

**Problem in Reference Paper:**
- Only 87.1% F1 for neuronal cells
- Severe class imbalance (180:1 ratio)
- Manual annotation introduces bias
- Unclear scVI configuration

**Our Solutions:**
1. SMOTE in embedding space (not raw 160K-dimensional data)
2. CellTypist on ALL genes before feature selection
3. Proper scVI training: exactly 100 epochs, no early stopping
4. Comprehensive evaluation: 180 models per dataset

---

## Datasets

### PBMC (Peripheral Blood Mononuclear Cells)
- **Source:** 10x Genomics multiome - PBMC 10K
- **Link:** https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-1-0-0
- **Cells:** 11,909 to 9,814 after QC
- **Types:** 8 immune cell types
- **Modalities:** RNA (26,240 genes) + ATAC (134,000 peaks)

### AD Neuronal (Alzheimer's Disease)
- **Source:** GEO Accession GSE214979
- **Link:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214979
- **Cells:** 65,778 to 5,000 stratified sample
- **Types:** 8 neuronal cell types
- **Challenge:** 180:1 class imbalance

---

## Detailed Results

### AD Neuronal Cells: Major Breakthrough

| Method | Modality | Classifier | F1 Score | vs. Baseline |
|--------|----------|------------|----------|--------------|
| **PCA** | RNA+ATAC | RandomForest | **99.96%** | **+14.7%** |
| **FA** | RNA+ATAC | SVM | 99.73% | +12.5% |
| **scVI** | RNA+ATAC | SVM | 98.82% | +11.6% |
| Baseline | RNA+ATAC | LogReg | 87.1% | - |

**All cell types >99% F1**, including rare classes:
- Pericytes (5 cells): 99.87% F1
- Endothelial (18 cells): 99.89% F1
- OPC (48 cells): 99.92% F1

### PBMC Immune Cells

| Method | Modality | Classifier | F1 Score | vs. Baseline |
|--------|----------|------------|----------|--------------|
| **FA** | RNA+ATAC | XGBoost | **93.13%** | -1.5% |
| **PCA** | RNA+ATAC | SVM | 92.94% | -1.7% |
| **scVI** | RNA+ATAC | SVM | 90.18% | -4.4% |
| Baseline | RNA+ATAC | SVM | 94.6% | - |

**Comparable performance** with fully automated pipeline.

### ATAC Contribution Analysis

| Dataset | RNA-only F1 | RNA+ATAC F1 | ATAC Benefit |
|---------|-------------|-------------|--------------|
| **AD Neuronal** | 99.16% | 99.96% | **+0.7%** |
| **PBMC Immune** | 92.88% | 93.13% | **+0.3%** |

**Key Insight:** ATAC provides minimal benefit when proper methodology is applied. Quality RNA-only data with good methods > Poor multi-modal data.

---

## Repository Structure

\`\`\`
improved-single-cell-annotation/
├── README.md                    # This file
├── Scripts/
│   ├── 01_data_preparation.py
│   ├── 02_2_ground_truth_labeling_RNA_ATAC.py
│   ├── 03_3_supervised_classification_scVI_RNA_ATAC.py
│   ├── 03_4_supervised_classification_PCA_RNA_ATAC.py
│   ├── 03_5_supervised_classification_FA_RNA_ATAC.py
│   └── pbmc_all_figures.py
├── Data/                        # (Ignored in git - too large)
│   ├── raw/
│   └── processed/
├── Results/                     # Summary CSVs and figures
│   ├── Metrics/
│   ├── Embeddings/
│   └── Figures/
├── Models/                      # (Ignored in git)
└── Logs/                        # (Ignored in git)
\`\`\`

---

## Technical Details

### Dependencies

**Core:**
- Python 3.10+
- scanpy 1.9.6
- muon 0.1.5
- scvi-tools 1.0.4
- celltypist 1.6.2

**ML:**
- scikit-learn 1.3.2
- xgboost 2.0.3
- imbalanced-learn 0.11.0

**Data & Viz:**
- numpy 1.24.3
- pandas 2.0.3
- matplotlib 3.7.1
- seaborn 0.12.2

### Runtime

- **Data Prep:** ~2 minutes
- **Annotation:** ~5 minutes
- **scVI Classification:** ~45 minutes per dataset
- **PCA/FA Classification:** ~20 minutes per dataset
- **Total:** ~1 hour per dataset for all methods

### Hardware Requirements

- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** ~50GB for data + models
- **CPU:** Multi-core recommended (scVI training is CPU-intensive on Mac)

---

## Critical Fixes Applied

### Fix #1: CellTypist Gene Matching
**Problem:** Only 794/2000 genes (39.7%) matched when using feature-selected data.  
**Solution:** Run CellTypist on ALL 26,000 genes BEFORE feature selection to 18,000+ matches (69%+).

### Fix #2: scVI Overfitting
**Problem:** scVI trained for 400 epochs due to early stopping bug.  
**Solution:** \`early_stopping=False\` forces exactly 100 epochs.

### Fix #3: Mac MPS Incompatibility
**Problem:** \`use_gpu=True\` on Mac M-series causes NaN losses.  
**Solution:** \`use_gpu=False\` for stable CPU-only training.

---

## Citation

If you use this code, please cite:

\`\`\`bibtex
@article{billah2025improved,
  title={Improved Single-Cell Multi-Modal Classification with Machine Learning},
  author={Billah, Arif},
  journal={Final Year Design Project},
  year={2025}
}
\`\`\`

**Reference Paper:**
\`\`\`bibtex
@article{gill2025combining,
  title={Combining single-cell ATAC and RNA sequencing for supervised cell annotation},
  author={Gill, Jaidip and Dasgupta, Abhijit and Manry, Brychan and Markuzon, Natasha},
  journal{BMC Bioinformatics},
  volume={26},
  number={67},
  year={2025}
}
\`\`\`

---

## Contact

**Arif Billah**  
- GitHub: [@arifbillah30](https://github.com/arifbillah30)

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **Reference Paper Authors:** Gill et al. (2025) for comprehensive methodology
- **10x Genomics:** Public PBMC multiome dataset
- **GSE214979 Authors:** AD neuronal dataset
- **Open-Source Community:** Scanpy, scvi-tools, CellTypist developers

---

**Keywords:** Single-cell multi-omics, Chromatin accessibility, RNA sequencing, ATAC-seq, Machine learning in genomics, Cell type annotation, SMOTE, Class imbalance, CellTypist, scVI
