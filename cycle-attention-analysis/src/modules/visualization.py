import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

class AttentionVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_all_plots(self, results, layer):
        """Create all visualization plots."""
        plots_dir = self.output_dir.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Cycle consistency comparison
        self.plot_cycle_consistency_comparison(results, layer, plots_dir)
        
        # 2. Attention entropy comparison
        self.plot_attention_entropy_comparison(results, layer, plots_dir)
        
        # 3. Head consistency heatmaps
        self.plot_head_consistency_heatmaps(results, layer, plots_dir)
        
        # 4. Attention pattern examples
        self.plot_attention_examples(results, layer, plots_dir)
        
        # 5. Summary statistics
        self.create_summary_report(results, layer, plots_dir)
    
    def plot_cycle_consistency_comparison(self, results, layer, output_dir):
        """Plot cycle consistency comparison across sequence types."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        consistency_data = []
        
        for seq_type in ['natural', 'icl']:
            if seq_type in results and results[seq_type] is not None:
                stats = results[seq_type]['head_statistics']
                for layer_name, layer_data in stats.items():
                    for head_name, head_stats in layer_data.items():
                        head_idx = int(head_name.split('_')[1])
                        consistency_data.append({
                            'sequence_type': seq_type,
                            'head': head_idx,
                            'consistency': head_stats['mean_consistency'],
                            'consistency_std': head_stats['std_consistency']
                        })
        
        if consistency_data:
            df = pd.DataFrame(consistency_data)
            
            # Box plot
            sns.boxplot(data=df, x='sequence_type', y='consistency', ax=axes[0])
            axes[0].set_title('Cycle Consistency by Sequence Type')
            axes[0].set_ylabel('Mean Consistency Score')
            
            # Scatter plot by head
            for seq_type in df['sequence_type'].unique():
                subset = df[df['sequence_type'] == seq_type]
                axes[1].scatter(subset['head'], subset['consistency'], 
                              label=seq_type, alpha=0.7)
            
            axes[1].set_xlabel('Attention Head')
            axes[1].set_ylabel('Mean Consistency Score')
            axes[1].set_title('Cycle Consistency by Head')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'cycle_consistency_layer_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_entropy_comparison(self, results, layer, output_dir):
        """Plot attention entropy comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        entropy_data = []
        
        for seq_type in ['natural', 'icl', 'no_cycle']:
            if seq_type in results and results[seq_type] is not None:
                stats = results[seq_type]['head_statistics']
                for layer_name, layer_data in stats.items():
                    for head_name, head_stats in layer_data.items():
                        head_idx = int(head_name.split('_')[1])
                        entropy_data.append({
                            'sequence_type': seq_type,
                            'head': head_idx,
                            'entropy': head_stats['mean_entropy'],
                            'entropy_std': head_stats['std_entropy']
                        })
        
        if entropy_data:
            df = pd.DataFrame(entropy_data)
            
            # Box plot
            sns.boxplot(data=df, x='sequence_type', y='entropy', ax=axes[0])
            axes[0].set_title('Attention Entropy by Sequence Type')
            axes[0].set_ylabel('Mean Entropy')
            
            # Violin plot
            sns.violinplot(data=df, x='sequence_type', y='entropy', ax=axes[1])
            axes[1].set_title('Attention Entropy Distribution')
            axes[1].set_ylabel('Mean Entropy')
            
            # Head comparison
            pivot_df = df.pivot_table(values='entropy', index='head', columns='sequence_type', fill_value=0)
            sns.heatmap(pivot_df.T, annot=True, fmt='.2f', ax=axes[2], cmap='viridis')
            axes[2].set_title('Entropy by Head and Sequence Type')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'attention_entropy_layer_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_head_consistency_heatmaps(self, results, layer, output_dir):
        """Create heatmaps showing consistency for each head."""
        seq_types = ['natural', 'icl']
        fig, axes = plt.subplots(len(seq_types), 1, figsize=(12, 4 * len(seq_types)))
        
        if len(seq_types) == 1:
            axes = [axes]
        
        for idx, seq_type in enumerate(seq_types):
            if seq_type in results and results[seq_type] is not None:
                stats = results[seq_type]['head_statistics']
                
                consistency_matrix = []
                head_names = []
                
                for layer_name, layer_data in stats.items():
                    layer_consistencies = []
                    layer_head_names = []
                    
                    for head_name in sorted(layer_data.keys(), key=lambda x: int(x.split('_')[1])):
                        head_stats = layer_data[head_name]
                        layer_consistencies.append(head_stats['mean_consistency'])
                        layer_head_names.append(f"{layer_name}_{head_name}")
                    
                    if layer_consistencies:
                        consistency_matrix.append(layer_consistencies)
                        head_names.extend(layer_head_names)
                
                if consistency_matrix:
                    # Convert to proper matrix format
                    max_heads = max(len(row) for row in consistency_matrix)
                    padded_matrix = []
                    for row in consistency_matrix:
                        padded_row = row + [0] * (max_heads - len(row))
                        padded_matrix.append(padded_row)
                    
                    head_labels = [f"Head {i}" for i in range(max_heads)]
                    
                    sns.heatmap(padded_matrix, 
                              xticklabels=head_labels,
                              yticklabels=[f"Layer {i}" for i in range(len(padded_matrix))],
                              annot=True, fmt='.2f', 
                              cmap='RdYlBu_r', ax=axes[idx])
                    axes[idx].set_title(f'Cycle Consistency Heatmap - {seq_type.upper()}')
            else:
                axes[idx].text(0.5, 0.5, f'No data for {seq_type}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'Cycle Consistency Heatmap - {seq_type.upper()}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'consistency_heatmaps_layer_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_examples(self, results, layer, output_dir):
        """Plot example attention patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for seq_type in ['natural', 'icl']:
            if seq_type in results and results[seq_type] is not None:
                data = results[seq_type]
                
                if data['attention_matrices']:
                    # Get first example
                    example = data['attention_matrices'][0]
                    
                    for layer_name, layer_data in example.items():
                        if plot_idx >= len(axes):
                            break
                        
                        # Get first head's attention matrix
                        first_head = list(layer_data.keys())[0]
                        attention_matrix = layer_data[first_head]['attention_matrix']
                        
                        # Plot attention heatmap
                        sns.heatmap(attention_matrix, 
                                  cmap='Blues', ax=axes[plot_idx],
                                  xticklabels=False, yticklabels=False)
                        axes[plot_idx].set_title(f'{seq_type.upper()} - {layer_name} - {first_head}')
                        plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'attention_examples_layer_{layer}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, results, layer, output_dir):
        """Create a text summary report."""
        report_path = output_dir / f'summary_report_layer_{layer}.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"Attention Analysis Summary - Layer {layer}\n")
            f.write("=" * 50 + "\n\n")
            
            for seq_type in ['natural', 'icl', 'no_cycle']:
                if seq_type in results and results[seq_type] is not None:
                    data = results[seq_type]
                    f.write(f"{seq_type.upper()} SEQUENCES:\n")
                    f.write("-" * 30 + "\n")
                    
                    f.write(f"Number of sequences analyzed: {len(data['sequence_info'])}\n")
                    
                    if 'head_statistics' in data:
                        stats = data['head_statistics']
                        
                        # Overall statistics
                        all_consistency = []
                        all_entropy = []
                        
                        for layer_name, layer_data in stats.items():
                            for head_name, head_stats in layer_data.items():
                                all_consistency.append(head_stats['mean_consistency'])
                                all_entropy.append(head_stats['mean_entropy'])
                        
                        if all_consistency:
                            f.write(f"Average cycle consistency: {np.mean(all_consistency):.3f} ± {np.std(all_consistency):.3f}\n")
                        if all_entropy:
                            f.write(f"Average attention entropy: {np.mean(all_entropy):.3f} ± {np.std(all_entropy):.3f}\n")
                        
                        # Top consistent heads
                        head_consistency = []
                        for layer_name, layer_data in stats.items():
                            for head_name, head_stats in layer_data.items():
                                head_idx = int(head_name.split('_')[1])
                                head_consistency.append((head_idx, head_stats['mean_consistency']))
                        
                        if head_consistency:
                            top_heads = sorted(head_consistency, key=lambda x: x[1], reverse=True)[:5]
                            f.write(f"Top 5 most consistent heads: {top_heads}\n")
                    
                    f.write("\n")
            
            f.write(f"\nAnalysis completed. Plots saved in: {output_dir.parent / 'plots'}\n")