import torch
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy

class AttentionAnalyzerFixed:
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    
    def analyze_attention_patterns(self, sequences, model, tokenizer, sequence_type):
        """Analyze attention patterns for a list of sequences using direct model calls."""
        if not sequences:
            return None
            
        results = {
            'attention_matrices': [],
            'attention_words': [],
            'cycle_consistency': [],
            'attention_entropy': [],
            'sequence_info': [],
            'head_statistics': {}
        }
        
        # Process sequences in batches
        batch_size = 8
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Analyzing {sequence_type}"):
            batch = sequences[i:i+batch_size]
            batch_results = self._process_batch_fixed(batch, model, tokenizer, sequence_type)
            
            # Accumulate results
            for key in ['attention_matrices', 'attention_words', 'cycle_consistency', 
                       'attention_entropy', 'sequence_info']:
                results[key].extend(batch_results[key])
        
        # Compute head statistics
        results['head_statistics'] = self._compute_head_statistics(results)
        
        return results
    
    def _process_batch_fixed(self, batch, model, tokenizer, sequence_type):
        """Process a batch using direct model forward pass to get attention weights."""
        
        # Prepare input tensors
        input_sequences = [seq['sequence'] for seq in batch]
        max_len = max(len(seq) for seq in input_sequences)
        
        # Pad sequences
        input_ids = []
        attention_masks = []
        
        for seq in input_sequences:
            if len(seq) > max_len:
                seq = seq[:max_len]  # Truncate if too long
            padded = [tokenizer.pad_token_id] * (max_len - len(seq)) + seq
            mask = [0] * (max_len - len(seq)) + [1] * len(seq)
            input_ids.append(padded)
            attention_masks.append(mask)
        
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_masks = torch.tensor(attention_masks, device=self.device)
        
        # Get attention weights using direct forward pass
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    output_attentions=True
                )
            
            attention_weights = outputs.attentions  # List of tensors, one per layer
            
        except Exception as e:
            print(f"Error getting attention weights: {e}")
            # Return empty results
            return {
                'attention_matrices': [{}] * len(batch),
                'attention_words': [{}] * len(batch),
                'cycle_consistency': [{}] * len(batch),
                'attention_entropy': [{}] * len(batch),
                'sequence_info': batch
            }
        
        batch_results = {
            'attention_matrices': [],
            'attention_words': [],
            'cycle_consistency': [],
            'attention_entropy': [],
            'sequence_info': []
        }
        
        # Process each sequence in the batch
        for seq_idx, seq_data in enumerate(batch):
            seq_results = self._analyze_sequence_attention_fixed(
                attention_weights, seq_idx, seq_data, tokenizer, sequence_type, max_len
            )
            
            for key in batch_results:
                batch_results[key].append(seq_results[key])
        
        return batch_results
    
    def _analyze_sequence_attention_fixed(self, attention_weights, seq_idx, seq_data, tokenizer, sequence_type, max_len):
        """Analyze attention patterns for a single sequence using actual attention weights."""
        
        results = {
            'attention_matrices': {},
            'attention_words': {},
            'cycle_consistency': {},
            'attention_entropy': {},
            'sequence_info': seq_data
        }
        
        # Get the actual sequence length (without padding)
        actual_seq_len = len(seq_data['sequence'])
        pad_len = max_len - actual_seq_len
        
        # Process each layer's attention
        for layer_idx, layer_attention in enumerate(attention_weights):
            layer_name = f"gpt_neox.layers.{layer_idx}"
            
            # Extract attention for this sequence: [batch, heads, seq_len, seq_len]
            if seq_idx < layer_attention.shape[0]:
                seq_attention = layer_attention[seq_idx]  # [heads, seq_len, seq_len]
                
                # Remove padding from attention weights
                if pad_len > 0:
                    # Remove padded positions
                    seq_attention = seq_attention[:, pad_len:, pad_len:]  # [heads, actual_len, actual_len]
                
                num_heads = seq_attention.shape[0]
                
                # Analyze each head
                layer_results = {}
                for head_idx in range(num_heads):
                    head_attention = seq_attention[head_idx].cpu().numpy()  # [seq_len, seq_len]
                    
                    head_results = self._analyze_head_attention_fixed(
                        head_attention, seq_data, tokenizer, sequence_type
                    )
                    
                    layer_results[f"head_{head_idx}"] = head_results
                
                results['attention_matrices'][layer_name] = layer_results
                
                # Aggregate layer-level results
                self._aggregate_layer_results(results, layer_name, layer_results)
        
        return results
    
    def _analyze_head_attention_fixed(self, head_attention, seq_data, tokenizer, sequence_type):
        """Analyze attention patterns for a single head with actual attention weights."""
        
        results = {
            'attention_matrix': head_attention,
            'attended_words': [],
            'cycle_consistency': 0.0,
            'entropy': 0.0
        }
        
        seq_len = head_attention.shape[0]
        tokens = seq_data['sequence']
        
        # Get words being attended to (for each position, find top attended tokens)
        for pos in range(seq_len):
            if pos < len(tokens):
                attention_weights = head_attention[pos, :pos+1]  # Causal attention
                if len(attention_weights) > 0:
                    # Get top 3 attended positions
                    top_indices = np.argsort(attention_weights)[-3:][::-1]
                    top_words = []
                    for idx in top_indices:
                        if idx < len(tokens):
                            word = tokenizer.decode([tokens[idx]]).strip()
                            weight = attention_weights[idx]
                            top_words.append((word, float(weight), int(idx)))
                    results['attended_words'].append(top_words)
                else:
                    results['attended_words'].append([])
            else:
                results['attended_words'].append([])
        
        # Calculate cycle consistency (for sequences with cycles)
        if 'cycle' in seq_data and sequence_type in ['natural', 'icl']:
            cycle_length = len(seq_data['cycle'])
            n_cycles = seq_data.get('n_cycles', 1)
            
            if n_cycles > 1 and cycle_length > 0:
                consistency = self._calculate_cycle_consistency_fixed(
                    head_attention, cycle_length, n_cycles, seq_data, sequence_type
                )
                results['cycle_consistency'] = consistency
        
        # Calculate attention entropy
        entropies = []
        for pos in range(seq_len):
            attention_weights = head_attention[pos, :pos+1]
            if len(attention_weights) > 1 and np.sum(attention_weights) > 0:
                # Normalize weights
                normalized_weights = attention_weights / np.sum(attention_weights)
                # Calculate entropy
                ent = entropy(normalized_weights + 1e-12)
                entropies.append(ent)
        
        results['entropy'] = float(np.mean(entropies)) if entropies else 0.0
        
        return results
    
    def _calculate_cycle_consistency_fixed(self, head_attention, cycle_length, n_cycles, seq_data, sequence_type):
        """Calculate consistency across cycles using actual attention weights."""
        
        if sequence_type == 'natural':
            prompt_length = seq_data.get('prompt_length', 0)
            cycle_start_idx = seq_data.get('cycle_start_idx', 0)
            cycle_start_pos = prompt_length + cycle_start_idx
        else:
            cycle_start_pos = 0
        
        if cycle_start_pos + cycle_length * n_cycles > head_attention.shape[0]:
            return 0.0
        
        # Extract attention patterns for each cycle
        cycle_patterns = []
        for cycle_idx in range(n_cycles):
            start_pos = cycle_start_pos + cycle_idx * cycle_length
            end_pos = start_pos + cycle_length
            
            if end_pos <= head_attention.shape[0]:
                # Get attention pattern for this cycle
                cycle_attention = head_attention[start_pos:end_pos, :]
                cycle_patterns.append(cycle_attention)
        
        if len(cycle_patterns) < 2:
            return 0.0
        
        # Calculate pairwise similarities between cycles
        similarities = []
        for i in range(len(cycle_patterns)):
            for j in range(i + 1, len(cycle_patterns)):
                pattern1 = cycle_patterns[i].flatten()
                pattern2 = cycle_patterns[j].flatten()
                
                if np.sum(pattern1) > 0 and np.sum(pattern2) > 0:
                    similarity = 1 - cosine(pattern1, pattern2)
                    similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _aggregate_layer_results(self, results, layer_name, layer_results):
        """Aggregate results across heads for a layer."""
        if layer_name not in results['attention_words']:
            results['attention_words'][layer_name] = {}
            results['cycle_consistency'][layer_name] = {}
            results['attention_entropy'][layer_name] = {}
        
        for head_name, head_results in layer_results.items():
            results['attention_words'][layer_name][head_name] = head_results['attended_words']
            results['cycle_consistency'][layer_name][head_name] = head_results['cycle_consistency']
            results['attention_entropy'][layer_name][head_name] = head_results['entropy']
    
    def _compute_head_statistics(self, results):
        """Compute statistics across all sequences for each head."""
        statistics = {}
        
        # Aggregate cycle consistency across sequences
        for seq_results in results['cycle_consistency']:
            for layer_name, layer_data in seq_results.items():
                if layer_name not in statistics:
                    statistics[layer_name] = {}
                
                for head_name, consistency in layer_data.items():
                    if head_name not in statistics[layer_name]:
                        statistics[layer_name][head_name] = {
                            'consistency_scores': [],
                            'entropy_scores': []
                        }
                    statistics[layer_name][head_name]['consistency_scores'].append(consistency)
        
        # Aggregate entropy across sequences
        for seq_results in results['attention_entropy']:
            for layer_name, layer_data in seq_results.items():
                for head_name, entropy_score in layer_data.items():
                    if layer_name in statistics and head_name in statistics[layer_name]:
                        statistics[layer_name][head_name]['entropy_scores'].append(entropy_score)
        
        # Compute summary statistics
        for layer_name in statistics:
            for head_name in statistics[layer_name]:
                consistency_scores = statistics[layer_name][head_name]['consistency_scores']
                entropy_scores = statistics[layer_name][head_name]['entropy_scores']
                
                statistics[layer_name][head_name].update({
                    'mean_consistency': float(np.mean(consistency_scores)) if consistency_scores else 0.0,
                    'std_consistency': float(np.std(consistency_scores)) if consistency_scores else 0.0,
                    'mean_entropy': float(np.mean(entropy_scores)) if entropy_scores else 0.0,
                    'std_entropy': float(np.std(entropy_scores)) if entropy_scores else 0.0,
                })
        
        return statistics