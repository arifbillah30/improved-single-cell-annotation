"""
Generate Comprehensive Visualizations for AD Neuronal Classification Results
============================================================================
Compares all 6 experiments: scVI, PCA, FA × (RNA+ATAC, RNA-only)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(scripts_dir)
os.chdir(project_dir)

print("="*80)
print("GENERATING AD NEURONAL CLASSIFICATION VISUALIZATIONS")
print("="*80)

# Create output directory
output_dir = 'Results/Figures/AD'
os.makedirs(output_dir, exist_ok=True)

# Load all results
results = []
methods = {
    'scVI_RNA_ATAC': ('scVI', 'RNA+ATAC'),
    'scVI_RNA_ONLY': ('scVI', 'RNA-only'),
    'PCA_RNA_ATAC': ('PCA', 'RNA+ATAC'),
    'PCA_RNA_ONLY': ('PCA', 'RNA-only'),
    'FA_RNA_ATAC': ('FA', 'RNA+ATAC'),
    'FA_RNA_ONLY': ('FA', 'RNA-only')
}

for method_name, (embedding, modality) in methods.items():
    summary_path = f'Results/Metrics/AD/{method_name}/classification_results_{embedding if embedding != "FA" else "FA"}_summary.csv'
    
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df['Embedding'] = embedding
        df['Modality'] = modality
        df['Method'] = f'{embedding}\n{modality}'
        results.append(df)
        print(f"✓ Loaded {method_name}")
    else:
        print(f"✗ Missing {method_name}")

# Combine all results
all_results = pd.concat(results, ignore_index=True)

# Rename columns for consistency
all_results = all_results.rename(columns={
    'Accuracy_mean': 'Accuracy',
    'F1_Macro_mean': 'F1_Score',
    'Precision_Macro_mean': 'Precision',
    'Recall_Macro_mean': 'Recall'
})

print(f"\nTotal results loaded: {len(all_results)} rows")
print(f"Classifiers: {', '.join([str(c) for c in all_results['Classifier'].unique() if pd.notna(c)])}")
print(f"Embeddings: {', '.join([str(e) for e in all_results['Embedding'].unique() if pd.notna(e)])}")

# ============================================================================
# Figure 1: Overall Performance Comparison (F1 Scores)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('AD Neuronal Cell Classification - Comprehensive Performance Analysis\n5,000 Cells | 10 Bootstraps',
             fontsize=16, fontweight='bold', y=0.995)

# Subplot 1: F1 Score by Method and Classifier
ax1 = axes[0, 0]
methods_order = ['scVI\nRNA+ATAC', 'scVI\nRNA-only', 'PCA\nRNA+ATAC', 'PCA\nRNA-only', 'FA\nRNA+ATAC', 'FA\nRNA-only']
classifiers = ['RandomForest', 'SVM', 'XGBoost']
colors = {'RandomForest': '#2E86AB', 'SVM': '#A23B72', 'XGBoost': '#F18F01'}

x_pos = np.arange(len(methods_order))
width = 0.25

for i, classifier in enumerate(classifiers):
    f1_scores = []
    for method in methods_order:
        data = all_results[(all_results['Method'] == method) & (all_results['Classifier'] == classifier)]
        if len(data) > 0:
            f1_scores.append(data['F1_Score'].values[0])
        else:
            f1_scores.append(0)
    
    ax1.bar(x_pos + i*width, f1_scores, width, label=classifier, color=colors[classifier], alpha=0.8)

ax1.set_xlabel('Method', fontweight='bold')
ax1.set_ylabel('F1 Score (Macro)', fontweight='bold')
ax1.set_title('F1 Scores Across Methods and Classifiers', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(methods_order, rotation=0, ha='center')
ax1.legend(title='Classifier')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.95, 1.0])
ax1.axhline(y=0.871, color='red', linestyle='--', linewidth=2, label='Reference Paper (87.1%)', alpha=0.7)

# Subplot 2: Best Performer per Method
ax2 = axes[0, 1]
best_f1 = []
for method in methods_order:
    method_data = all_results[all_results['Method'] == method]
    best_f1.append(method_data['F1_Score'].max())

bars = ax2.bar(methods_order, best_f1, color=['#2E86AB', '#5DADE2', '#A569BD', '#BB8FCE', '#F39C12', '#F8C471'], alpha=0.8)
ax2.set_ylabel('Best F1 Score', fontweight='bold')
ax2.set_xlabel('Method', fontweight='bold')
ax2.set_title('Best Performance per Method', fontsize=12, fontweight='bold')
ax2.set_xticklabels(methods_order, rotation=0, ha='center')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.95, 1.0])
ax2.axhline(y=0.871, color='red', linestyle='--', linewidth=2, label='Reference (87.1%)', alpha=0.7)

# Add value labels on bars
for bar, value in zip(bars, best_f1):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 3: RNA+ATAC vs RNA-only Comparison
ax3 = axes[1, 0]
embeddings = ['scVI', 'PCA', 'FA']
rna_atac_f1 = []
rna_only_f1 = []

for emb in embeddings:
    atac_data = all_results[(all_results['Embedding'] == emb) & (all_results['Modality'] == 'RNA+ATAC')]
    only_data = all_results[(all_results['Embedding'] == emb) & (all_results['Modality'] == 'RNA-only')]
    
    rna_atac_f1.append(atac_data['F1_Score'].max())
    rna_only_f1.append(only_data['F1_Score'].max())

x = np.arange(len(embeddings))
width = 0.35

bars1 = ax3.bar(x - width/2, rna_atac_f1, width, label='RNA+ATAC', color='#3498DB', alpha=0.8)
bars2 = ax3.bar(x + width/2, rna_only_f1, width, label='RNA-only', color='#E74C3C', alpha=0.8)

ax3.set_ylabel('Best F1 Score', fontweight='bold')
ax3.set_xlabel('Embedding Method', fontweight='bold')
ax3.set_title('RNA+ATAC vs RNA-only Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(embeddings)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0.95, 1.0])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)

# Subplot 4: Improvement over Reference Paper
ax4 = axes[1, 1]
reference_f1 = 0.871
improvements = []
method_labels = []

for method in methods_order:
    method_data = all_results[all_results['Method'] == method]
    best_f1 = method_data['F1_Score'].max()
    improvement = ((best_f1 - reference_f1) / reference_f1) * 100
    improvements.append(improvement)
    method_labels.append(method)

bars = ax4.barh(method_labels, improvements, color=['#2E86AB', '#5DADE2', '#A569BD', '#BB8FCE', '#F39C12', '#F8C471'], alpha=0.8)
ax4.set_xlabel('Improvement over Reference (%)', fontweight='bold')
ax4.set_title('Improvement vs Reference Paper (87.1% F1)', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Add value labels
for bar, value in zip(bars, improvements):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
             f'+{value:.1f}%',
             ha='left', va='center', fontsize=9, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_overall_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir}/01_overall_performance_comparison.png")

# ============================================================================
# Figure 2: Detailed Metrics Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('AD Neuronal Classification - Detailed Metrics Analysis',
             fontsize=16, fontweight='bold', y=0.995)

# Subplot 1: Accuracy Comparison
ax1 = axes[0, 0]
for i, classifier in enumerate(classifiers):
    acc_scores = []
    for method in methods_order:
        data = all_results[(all_results['Method'] == method) & (all_results['Classifier'] == classifier)]
        if len(data) > 0:
            acc_scores.append(data['Accuracy'].values[0])
        else:
            acc_scores.append(0)
    
    ax1.bar(x_pos + i*width, acc_scores, width, label=classifier, color=colors[classifier], alpha=0.8)

ax1.set_xlabel('Method', fontweight='bold')
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_title('Accuracy Across Methods', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(methods_order, rotation=0, ha='center')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.95, 1.0])

# Subplot 2: Precision Comparison
ax2 = axes[0, 1]
for i, classifier in enumerate(classifiers):
    prec_scores = []
    for method in methods_order:
        data = all_results[(all_results['Method'] == method) & (all_results['Classifier'] == classifier)]
        if len(data) > 0:
            prec_scores.append(data['Precision'].values[0])
        else:
            prec_scores.append(0)
    
    ax2.bar(x_pos + i*width, prec_scores, width, label=classifier, color=colors[classifier], alpha=0.8)

ax2.set_xlabel('Method', fontweight='bold')
ax2.set_ylabel('Precision (Macro)', fontweight='bold')
ax2.set_title('Precision Across Methods', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(methods_order, rotation=0, ha='center')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.95, 1.0])

# Subplot 3: Recall Comparison
ax3 = axes[1, 0]
for i, classifier in enumerate(classifiers):
    rec_scores = []
    for method in methods_order:
        data = all_results[(all_results['Method'] == method) & (all_results['Classifier'] == classifier)]
        if len(data) > 0:
            rec_scores.append(data['Recall'].values[0])
        else:
            rec_scores.append(0)
    
    ax3.bar(x_pos + i*width, rec_scores, width, label=classifier, color=colors[classifier], alpha=0.8)

ax3.set_xlabel('Method', fontweight='bold')
ax3.set_ylabel('Recall (Macro)', fontweight='bold')
ax3.set_title('Recall Across Methods', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(methods_order, rotation=0, ha='center')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0.95, 1.0])

# Subplot 4: All Metrics Heatmap (Best Performer)
ax4 = axes[1, 1]
best_results = []
for method in methods_order:
    method_data = all_results[all_results['Method'] == method]
    method_data = method_data.dropna(subset=['F1_Score'])
    if len(method_data) > 0:
        best_idx = method_data['F1_Score'].idxmax()
        best_row = method_data.loc[best_idx]
        best_results.append([
            best_row['Accuracy'],
            best_row['Precision'],
            best_row['Recall'],
            best_row['F1_Score']
        ])
    else:
        best_results.append([0, 0, 0, 0])

heatmap_data = np.array(best_results).T
im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.95, vmax=1.0)
ax4.set_xticks(np.arange(len(methods_order)))
ax4.set_yticks(np.arange(4))
ax4.set_xticklabels(methods_order, rotation=0, ha='center')
ax4.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
ax4.set_title('Best Performance Heatmap', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(4):
    for j in range(len(methods_order)):
        text = ax4.text(j, i, f'{heatmap_data[i, j]:.4f}',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')

plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_detailed_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/02_detailed_metrics_comparison.png")

# ============================================================================
# Figure 3: Summary Table
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Create summary table
summary_data = []
for method in methods_order:
    method_data = all_results[all_results['Method'] == method]
    for classifier in classifiers:
        clf_data = method_data[method_data['Classifier'] == classifier]
        if len(clf_data) > 0:
            row = clf_data.iloc[0]
            summary_data.append([
                method.replace('\n', ' '),
                classifier,
                f"{row['F1_Score']:.4f}",
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}"
            ])

table = ax.table(cellText=summary_data,
                colLabels=['Method', 'Classifier', 'F1 Score', 'Accuracy', 'Precision', 'Recall'],
                cellLoc='center',
                loc='center',
                colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#3498DB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')

plt.title('AD Neuronal Classification - Complete Results Summary\n5,000 Cells | 10 Bootstraps | 8 Cell Types',
          fontsize=14, fontweight='bold', pad=20)
plt.savefig(f'{output_dir}/03_results_summary_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/03_results_summary_table.png")

print("\n" + "="*80)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {output_dir}/")
print("\nGenerated files:")
print("  1. 01_overall_performance_comparison.png")
print("  2. 02_detailed_metrics_comparison.png")
print("  3. 03_results_summary_table.png")
print("\n" + "="*80)
