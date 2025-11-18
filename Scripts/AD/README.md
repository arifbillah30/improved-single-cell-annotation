# AD (Neuronal Cells) Classification Pipeline
## Alzheimer's Disease Dataset

This directory contains all scripts for running classification experiments on the AD neuronal cell dataset (GSE214979).

**Dataset:** 65,778 neuronal cells from 7 AD patients + 8 controls

## Quick Start - Run All Experiments

```bash
# Step 1: Data Preparation (QC filtering)
python Scripts/AD/01_data_preparation.py

# Step 2: Ground Truth Labeling (uses published annotations)
python Scripts/AD/02_1_ground_truth_labeling_RNA_only.py
python Scripts/AD/02_2_ground_truth_labeling_RNA_ATAC.py

# Step 3: Classification Experiments
# scVI (deep learning)
python Scripts/AD/03_3_supervised_classification_scVI_RNA_ONLY.py
python Scripts/AD/03_3_supervised_classification_scVI_RNA_ATAC.py

# PCA (linear)
python Scripts/AD/03_4_supervised_classification_PCA_RNA_ONLY.py
python Scripts/AD/03_4_supervised_classification_PCA_RNA_ATAC.py

# Factor Analysis (linear)
python Scripts/AD/03_5_supervised_classification_FA_RNA_ONLY.py
python Scripts/AD/03_5_supervised_classification_FA_RNA_ATAC.py

# Step 4: Generate Figures
python Scripts/AD/ad_all_figures.py
```

## Expected Results (Based on Reference Paper)

**Key Finding:** ATAC does NOT help neuronal cells (may even hurt performance)

| Method | RNA-only F1 | RNA+ATAC F1 | Difference |
|--------|-------------|-------------|------------|
| PCA-SVM | ~87.1% | ~81.2% | **-5.9%** (worse!) |
| scVI-LR | ~85-90% | ~85-90% | No significant difference |

**Why?**
- Neuronal cells have less epigenetic diversity than immune cells
- Slower, more constant cell types
- ATAC adds noise rather than signal

## File Structure

```
Scripts/AD/
├── 01_data_preparation.py                           # QC filtering
├── 02_1_ground_truth_labeling_RNA_only.py          # Annotate RNA-only
├── 02_2_ground_truth_labeling_RNA_ATAC.py          # Annotate RNA+ATAC
├── 03_3_supervised_classification_scVI_RNA_ONLY.py
├── 03_3_supervised_classification_scVI_RNA_ATAC.py
├── 03_4_supervised_classification_PCA_RNA_ONLY.py
├── 03_4_supervised_classification_PCA_RNA_ATAC.py
├── 03_5_supervised_classification_FA_RNA_ONLY.py
├── 03_5_supervised_classification_FA_RNA_ATAC.py
└── ad_all_figures.py                                # Generate visualizations

Data/processed/AD/
├── ad_filtered.h5mu                                 # After Step 1
├── ad_labeled_RNA_only.h5mu                         # After Step 2.1
└── ad_labeled_RNA_ATAC.h5mu                         # After Step 2.2

Results/Metrics/AD_*/
├── AD_scVI_RNA_only/
├── AD_scVI_RNA_ATAC/
├── AD_PCA_RNA_ONLY/
├── AD_PCA_RNA_ATAC/
├── AD_FA_RNA_ONLY/
└── AD_FA_RNA_ATAC/

Models/AD_*/
├── AD_scVI_RNA_only/
├── AD_scVI_RNA_ATAC/
├── AD_PCA_RNA_ONLY/
├── AD_PCA_RNA_ATAC/
├── AD_FA_RNA_ONLY/
└── AD_FA_RNA_ATAC/

Results/Figures/AD/
├── Performance/
└── Comparisons/
```

## Comparison with PBMC

After running both datasets, you can compare:

| Aspect | PBMC (Immune) | AD (Neuronal) |
|--------|---------------|---------------|
| **ATAC Benefit** | ✅ Helps (+1-3%) | ❌ Doesn't help (0% or negative) |
| **Best Method** | FA RNA+ATAC (93.13%) | Likely PCA or FA RNA-only |
| **Cell Types** | Dynamic, epigenetically diverse | Stable, less epigenetic diversity |
| **Use Case** | Multi-modal sequencing worth it | Save money, use RNA-only |

## Computational Requirements

- **Time per method:** ~45-60 minutes (10 bootstraps × 3 classifiers)
- **Total time:** ~6-8 hours for all experiments
- **Storage:** ~3-4 GB for all models
- **RAM:** 16 GB recommended

## Notes

1. **Annotations:** AD uses pre-existing cell type annotations from the published dataset (Morabito et al., 2021), not CellTypist
2. **Cell types:** Excitatory neurons, Inhibitory neurons, Oligodendrocytes, Astrocytes, Microglia, etc.
3. **Comparison:** This validates the paper's tissue-specificity finding - ATAC utility depends on biological context
