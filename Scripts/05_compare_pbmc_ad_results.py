"""
Comprehensive Comparison: PBMC vs AD Results
============================================
Compare improvements over reference paper for both datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)

print("="*80)
print("COMPREHENSIVE COMPARISON: PBMC vs AD NEURONAL CELLS")
print("="*80)

# Reference paper results (from Gill et al. 2025)
REFERENCE_RESULTS = {
    'PBMC': {
        'RNA+ATAC': 0.946,  # scVI-SVM best performer
        'RNA-only': 0.907,  # scVI-SVM
        'best_method': 'scVI-SVM',
        'improvement_with_atac': True,
        'best_cell_type': 'CD4 TEM',
        'best_cell_type_improvement': 0.112  # F1 improvement
    },
    'AD': {
        'RNA+ATAC': 0.871,  # No improvement
        'RNA-only': 0.871,  # Same as RNA+ATAC
        'best_method': 'LR-scVI',
        'improvement_with_atac': False,
        'note': 'ATAC provides no benefit for neuronal cells'
    }
}

# Load results
def load_dataset_results(dataset_name):
    results = []
    methods = {
        'scVI_RNA_ATAC': ('scVI', 'RNA+ATAC'),
        'scVI_RNA_only': ('scVI', 'RNA-only') if dataset_name == 'PBMC' else ('scVI', 'RNA-only'),
        'PCA_RNA_ATAC': ('PCA', 'RNA+ATAC'),
        'PCA_RNA_ONLY': ('PCA', 'RNA-only'),
        'FA_RNA_ATAC': ('FA', 'RNA+ATAC'),
        'FA_RNA_ONLY': ('FA', 'RNA-only')
    }
    
    for method_name, (embedding, modality) in methods.items():
        if dataset_name == 'AD':
            method_name = method_name.replace('_only', '_ONLY')
        
        summary_path = f'Results/Metrics/{dataset_name}/{method_name}/classification_results_{embedding if embedding != "FA" else "FA"}_summary.csv'
        
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df['Embedding'] = embedding
            df['Modality'] = modality
            df['Method'] = f'{embedding}-{modality}'
            df['Dataset'] = dataset_name
            results.append(df)
            print(f"✓ Loaded {dataset_name}/{method_name}")
        else:
            print(f"✗ Missing {dataset_name}/{method_name}")
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        # Rename columns for consistency
        combined = combined.rename(columns={
            'Accuracy_mean': 'Accuracy',
            'F1_Macro_mean': 'F1_Score',
            'Precision_Macro_mean': 'Precision',
            'Recall_Macro_mean': 'Recall'
        })
        return combined
    return pd.DataFrame()

# Load both datasets
pbmc_results = load_dataset_results('PBMC')
ad_results = load_dataset_results('AD')

print(f"\nPBMC results: {len(pbmc_results)} rows")
print(f"AD results: {len(ad_results)} rows")

# Combine all results
all_results = pd.concat([pbmc_results, ad_results], ignore_index=True)

# ============================================================================
# Analysis 1: Best Performance Comparison
# ============================================================================
print("\n" + "="*80)
print("BEST PERFORMANCE ANALYSIS")
print("="*80)

for dataset in ['PBMC', 'AD']:
    dataset_data = all_results[all_results['Dataset'] == dataset]
    print(f"\n{dataset} Dataset:")
    print(f"  Reference Paper (RNA+ATAC): {REFERENCE_RESULTS[dataset]['RNA+ATAC']:.3f}")
    print(f"  Reference Paper (RNA-only): {REFERENCE_RESULTS[dataset]['RNA-only']:.3f}")
    
    best_rna_atac = dataset_data[dataset_data['Modality'] == 'RNA+ATAC']['F1_Score'].max()
    best_rna_only = dataset_data[dataset_data['Modality'] == 'RNA-only']['F1_Score'].max()
    
    print(f"  Our Method (RNA+ATAC): {best_rna_atac:.4f}")
    print(f"  Our Method (RNA-only): {best_rna_only:.4f}")
    
    improvement_atac = ((best_rna_atac - REFERENCE_RESULTS[dataset]['RNA+ATAC']) / REFERENCE_RESULTS[dataset]['RNA+ATAC']) * 100
    improvement_only = ((best_rna_only - REFERENCE_RESULTS[dataset]['RNA-only']) / REFERENCE_RESULTS[dataset]['RNA-only']) * 100
    
    print(f"  Improvement (RNA+ATAC): +{improvement_atac:.2f}%")
    print(f"  Improvement (RNA-only): +{improvement_only:.2f}%")

# ============================================================================
# Figure 1: Side-by-Side Comparison
# ============================================================================
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Comprehensive Comparison: PBMC vs AD Neuronal Cells\nImproved Methodology vs Reference Paper (Gill et al. 2025)',
             fontsize=18, fontweight='bold', y=0.995)

# Subplot 1: Best F1 Scores Comparison
ax1 = fig.add_subplot(gs[0, :])
datasets = ['PBMC', 'AD']
x = np.arange(len(datasets))
width = 0.2

ref_rna_atac = [REFERENCE_RESULTS[d]['RNA+ATAC'] for d in datasets]
ref_rna_only = [REFERENCE_RESULTS[d]['RNA-only'] for d in datasets]

our_rna_atac = []
our_rna_only = []
for dataset in datasets:
    dataset_data = all_results[all_results['Dataset'] == dataset]
    our_rna_atac.append(dataset_data[dataset_data['Modality'] == 'RNA+ATAC']['F1_Score'].max())
    our_rna_only.append(dataset_data[dataset_data['Modality'] == 'RNA-only']['F1_Score'].max())

bars1 = ax1.bar(x - 1.5*width, ref_rna_atac, width, label='Reference: RNA+ATAC', color='#E74C3C', alpha=0.7)
bars2 = ax1.bar(x - 0.5*width, ref_rna_only, width, label='Reference: RNA-only', color='#C0392B', alpha=0.7)
bars3 = ax1.bar(x + 0.5*width, our_rna_atac, width, label='Our Method: RNA+ATAC', color='#3498DB', alpha=0.9)
bars4 = ax1.bar(x + 1.5*width, our_rna_only, width, label='Our Method: RNA-only', color='#2874A6', alpha=0.9)

ax1.set_ylabel('F1 Score (Macro)', fontweight='bold', fontsize=12)
ax1.set_title('Best Performance: Our Method vs Reference Paper', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.85, 1.0])

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: PBMC Detailed Comparison
ax2 = fig.add_subplot(gs[1, 0])
pbmc_methods = pbmc_results.groupby(['Embedding', 'Modality'])['F1_Score'].max().reset_index()
pbmc_methods = pbmc_methods.sort_values('F1_Score', ascending=True)

colors_pbmc = ['#3498DB' if 'RNA+ATAC' in mod else '#2874A6' for mod in pbmc_methods['Modality']]
bars = ax2.barh(range(len(pbmc_methods)), pbmc_methods['F1_Score'], color=colors_pbmc, alpha=0.8)
ax2.set_yticks(range(len(pbmc_methods)))
ax2.set_yticklabels([f"{row['Embedding']}\n{row['Modality']}" for _, row in pbmc_methods.iterrows()], fontsize=9)
ax2.set_xlabel('F1 Score', fontweight='bold')
ax2.set_title('PBMC: All Methods Performance', fontsize=12, fontweight='bold')
ax2.axvline(x=0.946, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Reference Best (94.6%)')
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)
ax2.set_xlim([0.85, 1.0])

# Add value labels
for i, (bar, value) in enumerate(zip(bars, pbmc_methods['F1_Score'])):
    ax2.text(value, i, f' {value:.4f}', va='center', fontsize=8, fontweight='bold')

# Subplot 3: AD Detailed Comparison
ax3 = fig.add_subplot(gs[1, 1])
ad_methods = ad_results.groupby(['Embedding', 'Modality'])['F1_Score'].max().reset_index()
ad_methods = ad_methods.sort_values('F1_Score', ascending=True)

colors_ad = ['#F39C12' if 'RNA+ATAC' in mod else '#D68910' for mod in ad_methods['Modality']]
bars = ax3.barh(range(len(ad_methods)), ad_methods['F1_Score'], color=colors_ad, alpha=0.8)
ax3.set_yticks(range(len(ad_methods)))
ax3.set_yticklabels([f"{row['Embedding']}\n{row['Modality']}" for _, row in ad_methods.iterrows()], fontsize=9)
ax3.set_xlabel('F1 Score', fontweight='bold')
ax3.set_title('AD Neuronal: All Methods Performance', fontsize=12, fontweight='bold')
ax3.axvline(x=0.871, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Reference (87.1%)')
ax3.legend(fontsize=9)
ax3.grid(axis='x', alpha=0.3)
ax3.set_xlim([0.85, 1.0])

# Add value labels
for i, (bar, value) in enumerate(zip(bars, ad_methods['F1_Score'])):
    ax3.text(value, i, f' {value:.4f}', va='center', fontsize=8, fontweight='bold')

# Subplot 4: Improvement Percentages
ax4 = fig.add_subplot(gs[1, 2])
improvements = []
labels = []

for dataset in ['PBMC', 'AD']:
    dataset_data = all_results[all_results['Dataset'] == dataset]
    best_f1 = dataset_data['F1_Score'].max()
    ref_f1 = REFERENCE_RESULTS[dataset]['RNA+ATAC']
    improvement = ((best_f1 - ref_f1) / ref_f1) * 100
    improvements.append(improvement)
    labels.append(f"{dataset}\n({ref_f1:.1%} → {best_f1:.2%})")

bars = ax4.barh(labels, improvements, color=['#3498DB', '#F39C12'], alpha=0.8)
ax4.set_xlabel('Relative Improvement (%)', fontweight='bold')
ax4.set_title('Overall Improvement vs Reference', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for bar, value in zip(bars, improvements):
    ax4.text(value, bar.get_y() + bar.get_height()/2.,
             f' +{value:.1f}%',
             ha='left', va='center', fontsize=11, fontweight='bold', color='darkgreen')

# Subplot 5: PBMC Classifier Comparison
ax5 = fig.add_subplot(gs[2, 0])
classifiers = ['RandomForest', 'SVM', 'XGBoost']
pbmc_clf = pbmc_results.groupby('Classifier')['F1_Score'].agg(['mean', 'max']).reset_index()

x_clf = np.arange(len(classifiers))
width_clf = 0.35

ax5.bar(x_clf - width_clf/2, [pbmc_clf[pbmc_clf['Classifier'] == c]['mean'].values[0] if len(pbmc_clf[pbmc_clf['Classifier'] == c]) > 0 else 0 for c in classifiers],
        width_clf, label='Mean F1', color='#5DADE2', alpha=0.8)
ax5.bar(x_clf + width_clf/2, [pbmc_clf[pbmc_clf['Classifier'] == c]['max'].values[0] if len(pbmc_clf[pbmc_clf['Classifier'] == c]) > 0 else 0 for c in classifiers],
        width_clf, label='Max F1', color='#2E86AB', alpha=0.8)

ax5.set_xlabel('Classifier', fontweight='bold')
ax5.set_ylabel('F1 Score', fontweight='bold')
ax5.set_title('PBMC: Classifier Performance', fontsize=12, fontweight='bold')
ax5.set_xticks(x_clf)
ax5.set_xticklabels(classifiers, rotation=15, ha='right')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim([0.85, 1.0])

# Subplot 6: AD Classifier Comparison
ax6 = fig.add_subplot(gs[2, 1])
ad_clf = ad_results.groupby('Classifier')['F1_Score'].agg(['mean', 'max']).reset_index()

ax6.bar(x_clf - width_clf/2, [ad_clf[ad_clf['Classifier'] == c]['mean'].values[0] if len(ad_clf[ad_clf['Classifier'] == c]) > 0 else 0 for c in classifiers],
        width_clf, label='Mean F1', color='#F8C471', alpha=0.8)
ax6.bar(x_clf + width_clf/2, [ad_clf[ad_clf['Classifier'] == c]['max'].values[0] if len(ad_clf[ad_clf['Classifier'] == c]) > 0 else 0 for c in classifiers],
        width_clf, label='Max F1', color='#F39C12', alpha=0.8)

ax6.set_xlabel('Classifier', fontweight='bold')
ax6.set_ylabel('F1 Score', fontweight='bold')
ax6.set_title('AD: Classifier Performance', fontsize=12, fontweight='bold')
ax6.set_xticks(x_clf)
ax6.set_xticklabels(classifiers, rotation=15, ha='right')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)
ax6.set_ylim([0.85, 1.0])

# Subplot 7: RNA+ATAC vs RNA-only Effectiveness
ax7 = fig.add_subplot(gs[2, 2])

atac_benefit = []
dataset_labels = []

for dataset in ['PBMC', 'AD']:
    dataset_data = all_results[all_results['Dataset'] == dataset]
    best_atac = dataset_data[dataset_data['Modality'] == 'RNA+ATAC']['F1_Score'].max()
    best_only = dataset_data[dataset_data['Modality'] == 'RNA-only']['F1_Score'].max()
    
    benefit = best_atac - best_only
    atac_benefit.append(benefit)
    dataset_labels.append(dataset)

colors_benefit = ['green' if b > 0.001 else 'gray' for b in atac_benefit]
bars = ax7.bar(dataset_labels, atac_benefit, color=colors_benefit, alpha=0.8)
ax7.set_ylabel('ATAC Benefit (ΔF1)', fontweight='bold')
ax7.set_title('ATAC Contribution Analysis', fontsize=12, fontweight='bold')
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, atac_benefit):
    label = f'+{value:.4f}' if value > 0 else f'{value:.4f}'
    color = 'darkgreen' if value > 0.001 else 'gray'
    ax7.text(bar.get_x() + bar.get_width()/2., value,
             label,
             ha='center', va='bottom' if value > 0 else 'top',
             fontsize=11, fontweight='bold', color=color)

# Add annotations
ax7.text(0, atac_benefit[0] * 0.5, 'ATAC helps\nimmune cells', 
         ha='center', va='center', fontsize=9, style='italic', color='darkgreen')
ax7.text(1, atac_benefit[1] * 0.5, 'ATAC neutral\nfor neurons', 
         ha='center', va='center', fontsize=9, style='italic', color='gray')

plt.savefig('Results/Figures/COMPREHENSIVE_COMPARISON_PBMC_vs_AD.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: Results/Figures/COMPREHENSIVE_COMPARISON_PBMC_vs_AD.png")

# ============================================================================
# Generate Summary Table
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUMMARY TABLE")
print("="*80)

summary_data = []

for dataset in ['PBMC', 'AD']:
    dataset_data = all_results[all_results['Dataset'] == dataset]
    
    # Best overall
    best_idx = dataset_data['F1_Score'].idxmax()
    best_row = dataset_data.loc[best_idx]
    
    # Best RNA+ATAC
    best_atac = dataset_data[dataset_data['Modality'] == 'RNA+ATAC']['F1_Score'].max()
    # Best RNA-only
    best_only = dataset_data[dataset_data['Modality'] == 'RNA-only']['F1_Score'].max()
    
    # Reference
    ref_atac = REFERENCE_RESULTS[dataset]['RNA+ATAC']
    ref_only = REFERENCE_RESULTS[dataset]['RNA-only']
    
    summary_data.append({
        'Dataset': dataset,
        'Reference RNA+ATAC': f"{ref_atac:.3f}",
        'Our RNA+ATAC': f"{best_atac:.4f}",
        'Improvement (RNA+ATAC)': f"+{((best_atac - ref_atac) / ref_atac * 100):.1f}%",
        'Reference RNA-only': f"{ref_only:.3f}",
        'Our RNA-only': f"{best_only:.4f}",
        'Improvement (RNA-only)': f"+{((best_only - ref_only) / ref_only * 100):.1f}%",
        'Best Method': f"{best_row['Embedding']}-{best_row['Classifier']}",
        'Best F1': f"{best_row['F1_Score']:.4f}",
        'ATAC Benefit': f"{best_atac - best_only:+.4f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('Results/COMPARISON_SUMMARY_PBMC_vs_AD.csv', index=False)
print(f"✓ Saved: Results/COMPARISON_SUMMARY_PBMC_vs_AD.csv")

# Print summary table
print("\n" + "="*80)
print("FINAL COMPARISON SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("✓ COMPREHENSIVE COMPARISON COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  • PBMC: {((our_rna_atac[0] - REFERENCE_RESULTS['PBMC']['RNA+ATAC']) / REFERENCE_RESULTS['PBMC']['RNA+ATAC'] * 100):.1f}% improvement (RNA+ATAC)")
print(f"  • AD: {((our_rna_atac[1] - REFERENCE_RESULTS['AD']['RNA+ATAC']) / REFERENCE_RESULTS['AD']['RNA+ATAC'] * 100):.1f}% improvement (RNA+ATAC)")
print(f"  • ATAC helps PBMC: +{atac_benefit[0]:.4f} F1")
print(f"  • ATAC neutral for AD: {atac_benefit[1]:+.4f} F1")
print("="*80)
