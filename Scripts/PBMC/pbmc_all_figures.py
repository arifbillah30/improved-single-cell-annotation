"""
Generate All Figures for FYDP Defense
======================================
Creates comprehensive visualizations of all results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
RESULTS_DIR = Path('Results/Metrics')
FIGURES_DIR = Path('Results/Figures/PBMC')  # Using PBMC dataset

def load_all_results():
    """Load all classification results"""
    results = {}
    
    methods = {
        'scVI_RNA_ONLY': 'Results/Metrics/scVI_RNA_only/classification_results_scVI_summary.csv',
        'scVI_RNA_ATAC': 'Results/Metrics/scVI_RNA_ATAC/classification_results_scVI_summary.csv',
        'PCA_RNA_ONLY': 'Results/Metrics/PCA_RNA_ONLY/classification_results_PCA_summary.csv',
        'PCA_RNA_ATAC': 'Results/Metrics/PCA_RNA_ATAC/classification_results_PCA_summary.csv',
        'FA_RNA_ONLY': 'Results/Metrics/FA_RNA_ONLY/classification_results_FA_summary.csv',
        'FA_RNA_ATAC': 'Results/Metrics/FA_RNA_ATAC/classification_results_FA_summary.csv',
    }
    
    for name, path in methods.items():
        try:
            # Try reading first line to determine format
            with open(path, 'r') as f:
                first_line = f.readline()
                second_line = f.readline()
            
            # If second line starts with ",mean,std" it's multi-level
            if second_line.strip().startswith(',mean,std') or second_line.strip().startswith('mean,std'):
                df = pd.read_csv(path, header=[0, 1], index_col=0)
                # Flatten multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
                results[name] = df
                print(f"✓ Loaded {name} (multi-level)")
            else:
                # Regular format
                df = pd.read_csv(path)
                results[name] = df
                print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    return results

def create_master_comparison_table(results):
    """Create master comparison table"""
    data = []
    
    for name, df in results.items():
        embedding = name.split('_')[0]  # scVI, PCA, or FA
        modality = 'RNA+ATAC' if 'ATAC' in name else 'RNA-only'
        
        # Check if it's the index-based format or column-based
        if df.index.name in ['classifier', 'Classifier'] or 'classifier' in str(df.index.name).lower():
            # Index-based format (PCA style)
            for classifier, row in df.iterrows():
                if pd.isna(classifier) or classifier == '':
                    continue
                data.append({
                    'Embedding': embedding,
                    'Modality': modality,
                    'Classifier': classifier,
                    'F1_Mean': row.get('f1_macro_mean', row.get('F1_Macro_mean', 0)),
                    'F1_Std': row.get('f1_macro_std', row.get('F1_Macro_std', 0)),
                    'Accuracy_Mean': row.get('accuracy_mean', row.get('Accuracy_mean', 0)),
                    'Accuracy_Std': row.get('accuracy_std', row.get('Accuracy_std', 0)),
                    'Training_Time': row.get('training_time_mean', row.get('Training_Time_mean', 0)),
                })
        else:
            # Column-based format
            for _, row in df.iterrows():
                classifier = row.get('Classifier', row.get('classifier', 'Unknown'))
                data.append({
                    'Embedding': embedding,
                    'Modality': modality,
                    'Classifier': classifier,
                    'F1_Mean': row.get('F1_Macro_mean', row.get('f1_macro_mean', 0)),
                    'F1_Std': row.get('F1_Macro_std', row.get('f1_macro_std', 0)),
                    'Accuracy_Mean': row.get('Accuracy_mean', row.get('accuracy_mean', 0)),
                    'Accuracy_Std': row.get('Accuracy_std', row.get('accuracy_std', 0)),
                    'Training_Time': row.get('Training_Time_mean', row.get('training_time_mean', 0)),
                })
    
    return pd.DataFrame(data)

def plot_f1_comparison_bar(df):
    """Bar chart: F1 scores across all methods"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    df['Method'] = df['Embedding'] + ' ' + df['Modality']
    df_pivot = df.pivot_table(index='Method', columns='Classifier', values='F1_Mean', aggfunc='mean')
    df_pivot_std = df.pivot_table(index='Method', columns='Classifier', values='F1_Std', aggfunc='mean')
    
    # Sort by best F1
    df_pivot['Best'] = df_pivot.max(axis=1)
    df_pivot = df_pivot.sort_values('Best', ascending=True)
    df_pivot = df_pivot.drop('Best', axis=1)
    
    # Plot
    df_pivot.plot(kind='barh', ax=ax, width=0.8, 
                  color=['#2ecc71', '#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=0.5)
    
    # Add reference line at 91.9% (paper's best)
    ax.axvline(x=0.919, color='red', linestyle='--', linewidth=2, 
               label='Paper Best (91.9%)', alpha=0.7)
    
    ax.set_xlabel('F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Classification Performance: F1 Score Comparison\n(All Methods × All Classifiers)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Classifier', fontsize=10, title_fontsize=11, 
              loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0.85, 0.96])
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=7, padding=3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Performance/f1_comparison_all_methods.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Performance/f1_comparison_all_methods.png")
    plt.close()

def plot_best_method_comparison(df):
    """Compare best classifier for each embedding × modality"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get best F1 for each method
    best_results = df.loc[df.groupby(['Embedding', 'Modality'])['F1_Mean'].idxmax()]
    best_results['Method'] = best_results['Embedding'] + '\n' + best_results['Modality']
    best_results = best_results.sort_values('F1_Mean')
    
    # Plot
    bars = ax.barh(best_results['Method'], best_results['F1_Mean'], 
                   color=['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#2ecc71'],
                   edgecolor='black', linewidth=1.5, height=0.6)
    
    # Add error bars
    ax.errorbar(best_results['F1_Mean'], range(len(best_results)), 
                xerr=best_results['F1_Std'], fmt='none', 
                ecolor='black', capsize=5, linewidth=1.5, alpha=0.7)
    
    # Add reference line
    ax.axvline(x=0.919, color='red', linestyle='--', linewidth=2.5, 
               label='Paper Best (91.9%)', alpha=0.8)
    
    # Labels
    ax.set_xlabel('F1 Score (Macro)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Method', fontsize=13, fontweight='bold')
    ax.set_title('Best Classifier Performance per Method\n(Comparison with Reference Paper)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)
    ax.set_xlim([0.87, 0.94])
    
    # Add value labels with classifier names
    for i, (idx, row) in enumerate(best_results.iterrows()):
        ax.text(row['F1_Mean'] + 0.001, i, 
                f"{row['F1_Mean']:.4f}\n({row['Classifier']})", 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Performance/best_method_comparison.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Performance/best_method_comparison.png")
    plt.close()

def plot_embedding_comparison(df):
    """Compare embeddings (average across classifiers)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by embedding and modality
    grouped = df.groupby(['Embedding', 'Modality'])[['F1_Mean', 'Training_Time']].mean().reset_index()
    
    # F1 comparison
    for modality in ['RNA-only', 'RNA+ATAC']:
        data = grouped[grouped['Modality'] == modality]
        ax1.plot(data['Embedding'], data['F1_Mean'], 
                marker='o', markersize=10, linewidth=2.5, 
                label=modality, alpha=0.8)
    
    ax1.axhline(y=0.919, color='red', linestyle='--', linewidth=2, 
                label='Paper Best', alpha=0.7)
    ax1.set_ylabel('F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Embedding Method', fontsize=12, fontweight='bold')
    ax1.set_title('Embedding Method Comparison\n(Average F1 Score)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0.88, 0.94])
    
    # Training time comparison
    for modality in ['RNA-only', 'RNA+ATAC']:
        data = grouped[grouped['Modality'] == modality]
        ax2.bar(data['Embedding'] + '\n' + modality.replace('-', '\n'), 
                data['Training_Time'], alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_title('Computational Efficiency\n(Average Training Time per Bootstrap)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Comparisons/embedding_method_comparison.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Comparisons/embedding_method_comparison.png")
    plt.close()

def plot_modality_comparison(df):
    """Compare RNA-only vs RNA+ATAC"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    embeddings = ['scVI', 'PCA', 'FA']
    
    for idx, embedding in enumerate(embeddings):
        ax = axes[idx]
        data = df[df['Embedding'] == embedding]
        
        rna_only = data[data['Modality'] == 'RNA-only'].groupby('Classifier')['F1_Mean'].mean()
        rna_atac = data[data['Modality'] == 'RNA+ATAC'].groupby('Classifier')['F1_Mean'].mean()
        
        x = np.arange(len(rna_only))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rna_only.values, width, 
                       label='RNA-only', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, rna_atac.values, width, 
                       label='RNA+ATAC', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{embedding} Embedding', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(rna_only.index, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.85, 0.95])
        
        # Add value labels
        ax.bar_label(bars1, fmt='%.3f', fontsize=7, padding=2)
        ax.bar_label(bars2, fmt='%.3f', fontsize=7, padding=2)
    
    fig.suptitle('Multi-modal Benefit: RNA-only vs RNA+ATAC', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Comparisons/rna_only_vs_rna_atac.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Comparisons/rna_only_vs_rna_atac.png")
    plt.close()

def plot_accuracy_vs_time(df):
    """Scatter plot: Accuracy vs Training Time"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'scVI': '#e74c3c', 'PCA': '#3498db', 'FA': '#2ecc71'}
    markers = {'RNA-only': 'o', 'RNA+ATAC': 's'}
    
    for embedding in df['Embedding'].unique():
        for modality in df['Modality'].unique():
            data = df[(df['Embedding'] == embedding) & (df['Modality'] == modality)]
            ax.scatter(data['Training_Time'], data['F1_Mean'], 
                      s=200, alpha=0.7, 
                      color=colors[embedding], 
                      marker=markers[modality],
                      label=f'{embedding} {modality}',
                      edgecolors='black', linewidth=1.5)
    
    ax.axhline(y=0.919, color='red', linestyle='--', linewidth=2, 
               label='Paper Best (91.9%)', alpha=0.7)
    
    ax.set_xlabel('Training Time (seconds per bootstrap)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Computational Cost\n(Higher & Left is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=9, loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Comparisons/accuracy_vs_time.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Comparisons/accuracy_vs_time.png")
    plt.close()

def plot_classifier_performance_heatmap(df):
    """Heatmap: Classifier performance across methods"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df['Method'] = df['Embedding'] + ' ' + df['Modality']
    pivot = df.pivot_table(index='Method', columns='Classifier', values='F1_Mean')
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                center=0.91, vmin=0.88, vmax=0.94,
                cbar_kws={'label': 'F1 Score (Macro)'},
                linewidths=1, linecolor='black',
                ax=ax, annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax.set_title('Classifier Performance Heatmap\n(F1 Score across All Methods)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Performance/classifier_performance_heatmap.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Performance/classifier_performance_heatmap.png")
    plt.close()

def plot_metric_comparison_radar(df):
    """Radar chart: Multi-metric comparison of best methods"""
    from math import pi
    
    # Get top 4 methods by F1
    best_methods = df.loc[df.groupby(['Embedding', 'Modality'])['F1_Mean'].idxmax()]
    best_methods = best_methods.nlargest(4, 'F1_Mean')
    
    categories = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Speed\n(Inverse Time)']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics to 0-1 scale
    for idx, row in best_methods.iterrows():
        values = [
            row['F1_Mean'],
            row['Accuracy_Mean'],
            row['F1_Mean'],  # Approximation for precision
            row['F1_Mean'],  # Approximation for recall
            1 / (row['Training_Time'] + 0.1)  # Inverse time (normalized)
        ]
        
        # Normalize to 0-1
        values = [(v - 0.85) / 0.15 for v in values[:4]] + [values[4] / max(1 / (best_methods['Training_Time'] + 0.1))]
        values += values[:1]  # Complete the circle
        
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=f"{row['Embedding']} {row['Modality']}", alpha=0.7)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Performance Comparison\n(Top 4 Methods)', 
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Performance/multi_metric_radar.png', 
                bbox_inches='tight', dpi=300)
    print("✓ Saved: Performance/multi_metric_radar.png")
    plt.close()

def create_summary_table(df):
    """Create publication-ready summary table"""
    # Get best results per method
    best = df.loc[df.groupby(['Embedding', 'Modality'])['F1_Mean'].idxmax()]
    best = best.sort_values('F1_Mean', ascending=False)
    
    # Format table
    table_data = []
    for idx, row in best.iterrows():
        table_data.append({
            'Method': f"{row['Embedding']} ({row['Modality']})",
            'Best Classifier': row['Classifier'],
            'F1 Score': f"{row['F1_Mean']:.4f} ± {row['F1_Std']:.4f}",
            'Accuracy': f"{row['Accuracy_Mean']:.4f} ± {row['Accuracy_Std']:.4f}",
            'Training Time (s)': f"{row['Training_Time']:.2f}",
        })
    
    summary_df = pd.DataFrame(table_data)
    summary_df.to_csv(FIGURES_DIR / 'Performance/summary_table.csv', index=False)
    print("✓ Saved: Performance/summary_table.csv")
    
    return summary_df

def main():
    print("="*80)
    print("GENERATING ALL FIGURES FOR FYDP DEFENSE")
    print("="*80)
    print()
    
    # Load results
    print("Loading results...")
    results = load_all_results()
    print()
    
    if not results:
        print("❌ No results found!")
        return
    
    # Create master dataframe
    print("Creating master comparison table...")
    df = create_master_comparison_table(results)
    print(f"✓ Master table created: {len(df)} rows")
    print()
    
    # Generate all figures
    print("Generating figures...")
    print()
    
    plot_f1_comparison_bar(df)
    plot_best_method_comparison(df)
    plot_embedding_comparison(df)
    plot_modality_comparison(df)
    plot_accuracy_vs_time(df)
    plot_classifier_performance_heatmap(df)
    plot_metric_comparison_radar(df)
    summary_df = create_summary_table(df)
    
    print()
    print("="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print()
    print("Output locations:")
    print(f"  Performance:     {FIGURES_DIR / 'Performance'}")
    print(f"  Comparisons:     {FIGURES_DIR / 'Comparisons'}")
    print()
    print("Summary Table:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
