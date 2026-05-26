import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class CycleEvolutionPlotter:
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    
    def analyze_cycle_evolution(self, sequences, model, sequence_type, target_layers=None, all_heads=False, max_cycles=None):
        """
        Analyze how attention patterns evolve across cycles.
        
        Args:
            sequences: List of sequence data with cycles
            model: The language model
            sequence_type: Type of sequences being analyzed
            target_layers: Specific layers to analyze (default: [5, 10, 15, 20])
            all_heads: If True, analyze all attention heads (default: False, analyzes heads [0, 4, 8, 12])
            max_cycles: Maximum number of cycles to analyze (default: None, use all available)
        
        Returns:
            Dict with evolution data for visualization
        """
        
        if not sequences:
            return None
        
        if target_layers is None:
            target_layers = [5, 10, 15, 20]  # Focus on key layers
            
        # Determine which heads to analyze
        if all_heads:
            target_heads = list(range(16))  # All 16 heads for pythia-1.4b
        else:
            target_heads = [0, 4, 8, 12]  # Representative heads
        
        print(f"Analyzing cycle evolution for {len(sequences)} {sequence_type} sequences...")
        
        evolution_data = {
            'sequences': [],
            'layers': target_layers,
            'heads': target_heads,
            'sequence_type': sequence_type,
            'max_cycles': max_cycles
        }
        
        for seq_idx, seq_data in enumerate(tqdm(sequences, desc=f"Processing {sequence_type}")):
            try:
                seq_evolution = self._analyze_single_sequence_evolution(
                    seq_data, model, target_layers, target_heads, max_cycles, seq_idx
                )
                if seq_evolution:
                    evolution_data['sequences'].append(seq_evolution)
            except Exception as e:
                print(f"Error analyzing sequence {seq_idx}: {e}")
                continue
        
        return evolution_data
    
    def _analyze_single_sequence_evolution(self, seq_data, model, target_layers, target_heads, max_cycles, seq_idx):
        """Analyze how attention evolves across cycles for a single sequence."""
        
        # Extract cycle information
        if 'cycle' not in seq_data or 'n_cycles' not in seq_data:
            return None
        
        cycle_tokens = seq_data['cycle']
        cycle_length = len(cycle_tokens)
        n_cycles = seq_data['n_cycles']
        
        # Apply max_cycles limit if specified
        if max_cycles is not None and n_cycles > max_cycles:
            n_cycles = max_cycles
        
        # if n_cycles < 2 or cycle_length < 2:
        #     return None
        
        sequence = seq_data['sequence']
        prompt_length = seq_data.get('prompt_length', 0)
        cycle_start_idx = seq_data.get('cycle_start_idx', 0)
        
        # Calculate actual cycle positions
        cycle_start_pos = prompt_length + cycle_start_idx if 'prompt_length' in seq_data else cycle_start_idx
        
        if cycle_start_pos + cycle_length * n_cycles > len(sequence):
            return None
        
        # Get attention weights
        input_ids = torch.tensor([sequence]).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True)
            attention_weights = outputs.attentions
        except Exception as e:
            print(f"Error getting attention for sequence: {e}")
            return None
        
        # Analyze evolution for target layers
        layer_results = {}
        
        for layer_idx in target_layers:
            if layer_idx >= len(attention_weights):
                continue
            
            layer_attention = attention_weights[layer_idx][0]  # [num_heads, seq_len, seq_len]
            
            # For target heads, track how attention evolves across cycles
            head_results = {}
            
            for head_idx in target_heads:
                if head_idx >= layer_attention.shape[0]:
                    continue
                    
                head_attention = layer_attention[head_idx].cpu().numpy()
                
                evolution_matrix = self._extract_cycle_evolution_matrix(
                    head_attention, cycle_start_pos, cycle_length, n_cycles
                )
                
                if evolution_matrix is not None:
                    head_results[f"head_{head_idx}"] = evolution_matrix
            
            if head_results:
                layer_results[f"layer_{layer_idx}"] = head_results
        
        return {
            'sequence_id': seq_idx,
            'cycle_info': {
                'tokens': cycle_tokens,
                'text': self.tokenizer.decode(cycle_tokens),
                'length': cycle_length,
                'n_cycles': n_cycles,
                'start_pos': cycle_start_pos
            },
            'layer_results': layer_results
        }
    
    def _extract_cycle_evolution_matrix(self, attention_matrix, cycle_start_pos, cycle_length, n_cycles):
        """
        Extract attention evolution matrix across cycles.
        
        Returns matrix of shape [n_cycles, cycle_length, cycle_length]
        where result[cycle_i, pos_j, pos_k] = attention from position j to position k in cycle i
        """
        
        if cycle_start_pos + cycle_length * n_cycles > attention_matrix.shape[0]:
            return None
        
        evolution_matrix = np.zeros((n_cycles, cycle_length, cycle_length))
        
        for cycle_idx in range(n_cycles):
            cycle_start = cycle_start_pos + cycle_idx * cycle_length
            cycle_end = cycle_start + cycle_length
            
            # Extract attention within this cycle
            for pos_in_cycle in range(cycle_length):
                abs_pos = cycle_start + pos_in_cycle
                
                # Get attention from this position to all positions within the cycle
                cycle_attention = attention_matrix[abs_pos, cycle_start:cycle_end]
                
                # Normalize to make it a proper distribution
                if np.sum(cycle_attention) > 0:
                    cycle_attention = cycle_attention / np.sum(cycle_attention)
                
                evolution_matrix[cycle_idx, pos_in_cycle, :] = cycle_attention
        
        return evolution_matrix
    
    def plot_cycle_evolution_heatmap(self, evolution_data, output_dir, max_examples=3):
        """
        Create heatmaps showing how attention evolves across cycles.
        """
        
        if not evolution_data or not evolution_data['sequences']:
            print("No evolution data to plot")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_type = evolution_data['sequence_type']
        target_layers = evolution_data['layers']
        
        # Plot for top sequences (those with most cycles)
        sequences = evolution_data['sequences']
        sequences.sort(key=lambda x: x['cycle_info']['n_cycles'], reverse=True)
        
        for seq_idx, seq_data in enumerate(sequences[:max_examples]):
            cycle_info = seq_data['cycle_info']
            cycle_text = cycle_info['text']
            n_cycles = cycle_info['n_cycles']
            cycle_length = cycle_info['length']
            
            # Create subplot for each layer
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for layer_idx, layer_name in enumerate([f"layer_{l}" for l in target_layers[:4]]):
                if layer_name not in seq_data['layer_results']:
                    continue
                
                ax = axes[layer_idx]
                
                # Find the head with most interesting evolution pattern
                layer_data = seq_data['layer_results'][layer_name]
                best_head = self._find_most_variable_head(layer_data)
                
                if best_head and best_head in layer_data:
                    evolution_matrix = layer_data[best_head]
                    
                    # Create the evolution heatmap
                    # Average attention from each position across cycles
                    avg_attention_by_cycle = np.mean(evolution_matrix, axis=1)  # [n_cycles, cycle_length]
                    
                    # Plot heatmap
                    im = ax.imshow(avg_attention_by_cycle, cmap='Blues', aspect='auto')
                    
                    ax.set_title(f'{layer_name} {best_head}\n"{cycle_text}"')
                    ax.set_xlabel('Position in Cycle')
                    ax.set_ylabel('Cycle Number')
                    ax.set_xticks(range(cycle_length))
                    ax.set_yticks(range(n_cycles))
                    ax.set_yticklabels([f'Cycle {i+1}' for i in range(n_cycles)])
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{layer_name} - No data')
            
            plt.suptitle(f'{sequence_type.title()} Sequence {seq_idx+1}: Attention Evolution\n'
                        f'Pattern: "{cycle_text}" × {n_cycles} cycles')
            plt.tight_layout()
            plt.savefig(output_dir / f'{sequence_type}_evolution_heatmap_{seq_idx+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_attention_focus_evolution(self, evolution_data, output_dir):
        """
        Create line plots showing how attention to each position evolves across cycles.
        """
        
        if not evolution_data or not evolution_data['sequences']:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_type = evolution_data['sequence_type']
        target_layers = evolution_data['layers']
        
        # Aggregate data across all sequences
        aggregated_data = self._aggregate_evolution_data(evolution_data)
        
        # Create plots for each layer
        for layer_name, layer_data in aggregated_data.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Plot top 4 heads with most variation
            head_names = sorted(layer_data.keys(), 
                              key=lambda h: np.var(layer_data[h]['mean_evolution']), 
                              reverse=True)[:4]
            
            for i, head_name in enumerate(head_names):
                ax = axes[i]
                head_data = layer_data[head_name]
                
                mean_evolution = head_data['mean_evolution']  # [n_cycles, cycle_length]
                std_evolution = head_data['std_evolution']
                
                n_cycles, cycle_length = mean_evolution.shape
                
                # Plot evolution of attention to each position
                colors = plt.cm.tab10(np.linspace(0, 1, cycle_length))
                
                for pos in range(cycle_length):
                    mean_vals = mean_evolution[:, pos]
                    std_vals = std_evolution[:, pos]
                    
                    cycles = list(range(1, n_cycles + 1))
                    
                    ax.plot(cycles, mean_vals, 'o-', color=colors[pos], 
                           label=f'Position {pos}', alpha=0.8)
                    ax.fill_between(cycles, mean_vals - std_vals, mean_vals + std_vals, 
                                   color=colors[pos], alpha=0.2)
                
                ax.set_title(f'{layer_name} {head_name}')
                ax.set_xlabel('Cycle Number')
                ax.set_ylabel('Average Attention Weight')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{sequence_type.title()}: Attention Focus Evolution by Position')
            plt.tight_layout()
            plt.savefig(output_dir / f'{sequence_type}_focus_evolution_{layer_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _find_most_variable_head(self, layer_data):
        """Find the head with the most variation across cycles."""
        
        max_variance = 0
        best_head = None
        
        for head_name, evolution_matrix in layer_data.items():
            # Calculate variance across cycles for each position
            cycle_averages = np.mean(evolution_matrix, axis=2)  # Average attention per cycle
            total_variance = np.var(cycle_averages)
            
            if total_variance > max_variance:
                max_variance = total_variance
                best_head = head_name
        
        return best_head
    
    def _aggregate_evolution_data(self, evolution_data):
        """Aggregate evolution data across sequences for statistical analysis."""
        
        aggregated = {}
        
        for seq_data in evolution_data['sequences']:
            for layer_name, layer_data in seq_data['layer_results'].items():
                if layer_name not in aggregated:
                    aggregated[layer_name] = {}
                
                for head_name, evolution_matrix in layer_data.items():
                    if head_name not in aggregated[layer_name]:
                        aggregated[layer_name][head_name] = {
                            'matrices': [],
                            'mean_evolution': None,
                            'std_evolution': None
                        }
                    
                    # Average attention across the "to" dimension to get attention focus
                    attention_focus = np.mean(evolution_matrix, axis=2)  # [n_cycles, cycle_length]
                    aggregated[layer_name][head_name]['matrices'].append(attention_focus)
        
        # Compute statistics - handle variable-length sequences
        for layer_name in aggregated:
            for head_name in aggregated[layer_name]:
                matrices = aggregated[layer_name][head_name]['matrices']
                if matrices:
                    # Check if all matrices have the same shape
                    shapes = [m.shape for m in matrices]
                    if len(set(shapes)) == 1:
                        # All same shape - can stack normally
                        stacked = np.stack(matrices, axis=0)  # [n_sequences, n_cycles, cycle_length]
                        aggregated[layer_name][head_name]['mean_evolution'] = np.mean(stacked, axis=0)
                        aggregated[layer_name][head_name]['std_evolution'] = np.std(stacked, axis=0)
                    else:
                        # Different shapes - pad to max dimensions and mask
                        max_shape = tuple(max(dim) for dim in zip(*shapes))
                        padded_matrices = []
                        for matrix in matrices:
                            padded = np.full(max_shape, np.nan)
                            padded[:matrix.shape[0], :matrix.shape[1]] = matrix
                            padded_matrices.append(padded)
                        
                        stacked = np.stack(padded_matrices, axis=0)
                        # Use nanmean/nanstd to handle NaN values from padding
                        aggregated[layer_name][head_name]['mean_evolution'] = np.nanmean(stacked, axis=0)
                        aggregated[layer_name][head_name]['std_evolution'] = np.nanstd(stacked, axis=0)
        
        return aggregated
    
    def create_summary_plot(self, evolution_data, output_dir):
        """Create a summary plot showing overall trends."""
        
        if not evolution_data or not evolution_data['sequences']:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sequence_type = evolution_data['sequence_type']
        
        # Calculate overall consistency metrics
        consistency_by_layer = {}
        
        for seq_data in evolution_data['sequences']:
            for layer_name, layer_data in seq_data['layer_results'].items():
                if layer_name not in consistency_by_layer:
                    consistency_by_layer[layer_name] = []
                
                for head_name, evolution_matrix in layer_data.items():
                    # Calculate how consistent attention patterns are across cycles
                    consistency = self._calculate_evolution_consistency(evolution_matrix)
                    consistency_by_layer[layer_name].append(consistency)
        
        # Plot consistency by layer
        layer_names = sorted(consistency_by_layer.keys(), key=lambda x: int(x.split('_')[1]))
        layer_nums = [int(name.split('_')[1]) for name in layer_names]
        mean_consistencies = [np.mean(consistency_by_layer[name]) for name in layer_names]
        std_consistencies = [np.std(consistency_by_layer[name]) for name in layer_names]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(layer_nums, mean_consistencies, yerr=std_consistencies, 
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('Layer Number')
        plt.ylabel('Attention Evolution Consistency')
        plt.title(f'{sequence_type.title()}: How Consistent is Attention Across Cycles?\n'
                 f'(Higher = More Consistent, Lower = More Variable)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{sequence_type}_evolution_consistency_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary: {sequence_type} sequences show consistency range {np.min(mean_consistencies):.3f} to {np.max(mean_consistencies):.3f}")
    
    def _calculate_evolution_consistency(self, evolution_matrix):
        """Calculate how consistent attention patterns are across cycles."""
        
        n_cycles = evolution_matrix.shape[0]
        if n_cycles < 2:
            return 0.0
        
        # For each position in the cycle, calculate correlation between cycles
        correlations = []
        
        for pos in range(evolution_matrix.shape[1]):
            attention_to_pos = evolution_matrix[:, pos, :]  # [n_cycles, cycle_length]
            
            # Calculate pairwise correlations between cycles
            pair_corrs = []
            for i in range(n_cycles):
                for j in range(i + 1, n_cycles):
                    corr = np.corrcoef(attention_to_pos[i], attention_to_pos[j])[0, 1]
                    if not np.isnan(corr):
                        pair_corrs.append(corr)
            
            if pair_corrs:
                correlations.append(np.mean(pair_corrs))
        
        return np.mean(correlations) if correlations else 0.0