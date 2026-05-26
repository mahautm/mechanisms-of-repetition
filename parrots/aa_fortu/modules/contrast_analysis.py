import torch
import numpy as np
import torch
import numpy as np
from torch.amp import autocast
from warnings import warn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import re
from parrots.cycle_detection import detect_cycles

def compute_direct_contrasts(hooked_model, expected_next, lens, device, batch_idx, cycle_start_positions=None, original_sequence_lengths=None):
    """
    Compute contrasts directly from hooked model outputs.
    
    Args:
        hooked_model: The model with hooks that captured outputs
        expected_next: List of expected next tokens
        lens: Lens for projecting attention to logit space (single lens or dict)
        device: Device to run computations on
        batch_idx: Batch index for logging
        cycle_start_positions: List of cycle start positions for each sequence (optional)
        original_sequence_lengths: List of original input lengths before generation (optional)
        
    Returns:
        tuple: (acts_cyc, next_highest) dictionaries with results
    """
    acts_cyc = {}
    next_highest = {}
    
    # Step 1: Get final layer logits from the hooked model
    all_logits = hooked_model.get_all_logits()  # Get ALL logits from generation, not just final
    if all_logits is None:
        print(f"Warning: No final logits captured for batch {batch_idx}")
        return acts_cyc, next_highest
        
    # Step 2: Determine which positions to analyze
    if cycle_start_positions is not None and original_sequence_lengths is not None:
        # Use cycle start positions (these are relative to the GENERATED tokens only)
        # The all_logits tensor contains logits for generated tokens only, not original input
        analysis_positions = cycle_start_positions  # Use directly since they're relative to generation
        
        # Extract logits at cycle start positions
        valid_positions = [pos for pos in analysis_positions if 0 <= pos < all_logits.shape[1]]
        if not valid_positions:
            print(f"Warning: No valid cycle start positions for batch {batch_idx}")
            print(f"Positions requested: {analysis_positions}, available range: 0-{all_logits.shape[1]-1}")
            return acts_cyc, next_highest
            
        # Get logits at cycle start positions for each sequence
        last_logits = torch.zeros(len(analysis_positions), all_logits.shape[2], device=device)
        for i, pos in enumerate(analysis_positions):
            if 0 <= pos < all_logits.shape[1]:
                last_logits[i] = all_logits[i, pos, :]
            else:
                print(f"Warning: Position {pos} out of bounds for sequence {i}, using last available position")
                last_logits[i] = all_logits[i, -1, :]  # Fallback to last position
                
    else:
        # Fallback to last position analysis (original behavior)
        print(f"Batch {batch_idx}: No cycle positions provided, using last token positions")
        analysis_positions = [all_logits.shape[1] - 1] * all_logits.shape[0]
        last_logits = all_logits[:, -1, :]  # [batch, vocab_size]
            
    # Step 3: Find the second most likely token (next_highest) from the analyzed positions
    expected_next_tensor = torch.tensor(expected_next, device=device) if expected_next else None
    
    if expected_next_tensor is not None and len(expected_next) > 0:
        # Find highest and second highest tokens
        # Note: last_logits is now [batch, vocab_size] after position extraction
        top2_tokens = last_logits.topk(2, dim=-1).indices  # [batch, 2]
        highest_toks = top2_tokens[:, 0]  # Most likely tokens
        second_highest_toks = top2_tokens[:, 1]  # Second most likely tokens
        
        # Create mask for sequences where model prediction matches expected
        mask = highest_toks[:len(expected_next_tensor)] == expected_next_tensor
        
        # For sequences where prediction matches expected, use second highest as contrast
        # For sequences where prediction doesn't match, use the highest as contrast
        next_highest_tokens = torch.clone(highest_toks[:len(expected_next_tensor)])
        next_highest_tokens[mask] = second_highest_toks[:len(expected_next_tensor)][mask]
        
        print(f"Batch {batch_idx}: Found {mask.sum().item()}/{len(expected_next_tensor)} correct predictions")
        n_successful = mask.sum().item()
        # Step 3: Get outputs from hooked model (attention or MLP)
        if hooked_model.has_mlp_outputs():
            # Use MLP outputs if available
            outputs_to_process = hooked_model.get_mlp_outputs()
            output_type = "MLP"
        else:
            # Fall back to attention outputs
            outputs_to_process = hooked_model.attn_outputs
            output_type = "attention"
            
        if not outputs_to_process:
            print(f"Warning: No {output_type} outputs captured for batch {batch_idx}")
            return acts_cyc, next_highest, n_successful
            
        # Step 4: Process each output layer
        for layer_output in outputs_to_process:
            if layer_output is None:
                continue
            layer_name, layer_data = layer_output
            
            # Handle different output formats (attention heads vs MLP)
            if output_type == "MLP":
                # For MLP: typically [batch, seq, mlp_dim] - no head splitting needed
                if hasattr(layer_data, 'shape') and len(layer_data.shape) == 3:
                    # MLP output: create single "head" for consistency
                    attention_heads = [layer_data]  # Treat MLP as single head
                    print(f"Batch {batch_idx}, {layer_name}: Processing MLP output as single head")
                else:
                    print(f"Unexpected MLP tensor shape: {layer_data.shape}")
                    continue
            else:
                # Original attention head processing logic
                attention_heads = None
                
                if isinstance(layer_data, (list, tuple)) and len(layer_data) > 0:
                    # If it's a tuple/list, the first element is usually the attention output
                    if hasattr(layer_data[0], 'shape'):
                        layer_data = layer_data[0]  # Extract the actual attention tensor
                    else:
                        # If the first element doesn't have shape, treat the whole list as pre-split heads
                        attention_heads = layer_data
            
            # If we don't have pre-split heads, split the tensor ourselves (only for attention, not MLP)
            if output_type != "MLP" and attention_heads is None and hasattr(layer_data, 'shape') and len(layer_data.shape) >= 3:
                # Split tensor into individual heads
                if len(layer_data.shape) == 4:  # [batch, seq, heads, head_dim]
                    attention_heads = [layer_data[:, :, h, :] for h in range(layer_data.shape[2])]
                    print(f"Batch {batch_idx}, {layer_name}: Split 4D tensor into {len(attention_heads)} heads")
                elif len(layer_data.shape) == 3:  # [batch, seq, hidden_size] - most common case
                    # Split into individual attention heads for head-specific analysis
                    batch_size, seq_len, hidden_size = layer_data.shape
                    
                    # For pythia models, determine number of heads from hidden size
                    if hidden_size == 2048:
                        num_heads = 16
                    elif hidden_size == 4096:
                        num_heads = 32
                    elif hidden_size == 1024:
                        num_heads = 8
                    elif hidden_size == 768:
                        num_heads = 12
                    elif hidden_size == 512:
                        num_heads = 8  # Pythia-70m: 512/8 = 64 head_dim
                    else:
                        # Fallback: assume head_dim = 64 or 128
                        num_heads = hidden_size // 128 if hidden_size % 128 == 0 else hidden_size // 64
                        if num_heads <= 0:
                            num_heads = 1  # Fallback to single head
                    
                    head_dim = hidden_size // num_heads
                    
                    # Reshape to [batch, seq, num_heads, head_dim]
                    reshaped = layer_data.reshape(batch_size, seq_len, num_heads, head_dim)
                    # Split into individual heads: [batch, seq, head_dim] for each head
                    attention_heads = [reshaped[:, :, h, :] for h in range(num_heads)]
                else:
                    print(f"Unsupported attention tensor shape: {layer_data.shape}")
                    continue
            
            if attention_heads is None:
                print(f"Batch {batch_idx}, {layer_name}: Could not extract attention heads")
                continue
                                
            # Step 5: For each attention head, compute contrast
            for head_idx, head_attention in enumerate(attention_heads):
                if head_attention is None:
                    continue
                head_key = f"{layer_name}_head_{head_idx}"
                
                # Only process sequences with valid expected tokens
                if len(expected_next) == 0:
                    continue
                    
                # Filter attention by sequences that have expected tokens
                valid_seq_count = min(head_attention.shape[0], len(expected_next))
                head_attention_filtered = head_attention[:valid_seq_count]
                expected_filtered = expected_next_tensor[:valid_seq_count]
                next_highest_filtered = next_highest_tokens[:valid_seq_count]
                
                if head_attention_filtered.shape[0] == 0:
                    continue
                    
                # Step 6: Apply lens to project attention to logit space
                projected_logits = None  # Initialize before checking lens
                if lens is not None:
                    head_attention_filtered = head_attention_filtered.to(device)
                    
                    # For attention analysis, use the last position (standard approach)
                    # Note: Attention tensor contains full sequence (original + generated)
                    # Using last position gives us the attention at the end of generation
                    if len(head_attention_filtered.shape) == 3:  # [batch, seq, head_dim]
                        head_attention_filtered = head_attention_filtered[:, -1, :]  # Take last position
                    

                    # Step 6: Project attention to logit space
                    projected_logits = None  # Initialize to catch errors
                    if lens is not None:
                        # Handle different lens formats
                        if isinstance(lens, dict):
                            # Extract layer number from layer_name
                            layer_match = re.search(r'layers\.(\d+)', layer_name)
                            if layer_match:
                                layer_num = int(layer_match.group(1))
                                if layer_num in lens:
                                    current_lens = lens[layer_num].to(device).to(head_attention_filtered.dtype)
                                    
                                    # Check if it's a multi-head lens
                                    if hasattr(current_lens, 'head_lenses'):
                                        # Multi-head lens: extract head index from head_key
                                        head_match = re.search(r'head_(\d+)', head_key)
                                        if head_match:
                                            head_idx = int(head_match.group(1))
                                            projected_logits = current_lens(head_attention_filtered, head_idx=head_idx)
                                        else:
                                            print(f"Warning: Could not extract head index from {head_key}")
                                            continue
                                    else:
                                        # Single lens for the whole layer
                                        projected_logits = current_lens(head_attention_filtered)
                                else:
                                    print(f"Warning: No lens found for layer {layer_num}, only layers {list(lens.keys())} available")
                                    # Fall back to unembedding
                                    projected_logits = None
                            else:
                                print(f"Warning: Could not extract layer number from {layer_name}")
                                # Fall back to unembedding
                                projected_logits = None
                        else:
                            # Single lens object
                            current_lens = lens.to(device).to(head_attention_filtered.dtype)
                            
                            # Check if it's a multi-head lens
                            if hasattr(current_lens, 'head_lenses'):
                                # Multi-head lens: extract head index
                                head_match = re.search(r'head_(\d+)', head_key)
                                if head_match:
                                    head_idx = int(head_match.group(1))
                                    projected_logits = current_lens(head_attention_filtered, head_idx=head_idx)
                                else:
                                    print(f"Warning: Could not extract head index from {head_key}")
                                    # Fall back to unembedding
                                    projected_logits = None
                            else:
                                # Regular single lens
                                projected_logits = current_lens(head_attention_filtered)
                    
                    # If lens was not available or failed, use unembedding layer directly
                    if projected_logits is None:
                        # Use unembedding layer directly if no lens
                        # Ensure dtype compatibility
                        head_attention_filtered = head_attention_filtered.to(hooked_model.unembed_module.weight.dtype)
                        projected_logits = hooked_model.unembed(head_attention_filtered)
                    
                # Step 7: Compute probabilities for expected and next_highest tokens
                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(projected_logits.float(), dim=-1)
                
                # Get probabilities for expected tokens using advanced indexing
                # probs[i, expected_filtered[i]] gives probability of expected token for sequence i
                batch_indices = torch.arange(len(expected_filtered), device=device)
                prob_expected = probs[batch_indices, expected_filtered].cpu().detach().numpy()
                
                # Get probabilities for next highest tokens using advanced indexing
                prob_next_highest = probs[batch_indices, next_highest_filtered].cpu().detach().numpy()
                
                # Step 8: Store results for heatmap computation
                if head_key not in acts_cyc:
                    acts_cyc[head_key] = []
                    next_highest[head_key] = []
                    
                acts_cyc[head_key].append(prob_expected)
                next_highest[head_key].append(prob_next_highest)
                
    
    return acts_cyc, next_highest, n_successful

def extract_contrasts(text, hooked_model, tokenizer, lens=None, n_cycles=0, batch_size=1, max_length=256, max_new_tokens=100, no_head_analysis=False, layers=None):
    """
    Batch process text to extract contrastive head activations for both natural and ICL settings.
    All tensors are moved to the correct device before any operation.
    Returns: heatmap_dict, total_natural, icl_heatmap_dict, total_icl, data_index, rep_index
    """
    acts_cyc, next_highest = {}, {}
    icl_acts, icl_next_highest = {}, {}
    no_cycle_icl_acts, no_cycle_icl_next_highest = {}, {}
    data_index, rep_index, no_cycle_index = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = min(batch_size, len(text))
    total_natural, total_icl, total_no_cycle = 0, 0, 0
    total_successful_nat, total_successful_icl, total_successful_no_cycle = 0, 0, 0
    # Set padding side for decoder-only models
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pre-tokenize all text in advance for speed
    pretokenized = [tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=max_length) for t in text]
    for i in tqdm(range(0, len(text), batch_size), desc="analysing samples", total=(len(text) + batch_size - 1)//batch_size):
        batch = pretokenized[i:i+batch_size]
        # Pad input_ids and attention_mask to the same length before concatenation
        input_ids_list = [b['input_ids'].squeeze(0) for b in batch]
        attention_mask_list = [b['attention_mask'].squeeze(0) for b in batch]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        toked = {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)}
        
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.float):
            o1 = hooked_model.generate(**toked, do_sample=False, max_new_tokens=max_new_tokens)
        # 1. Find prompt lengths
        plengths = toked["attention_mask"].sum(dim=1).tolist()
        # 2. Ensure o1 and all tensors are on CPU for detect_cycles (if needed)
        o1_cpu = [o1[j].detach().cpu() if o1[j].device.type != 'cpu' else o1[j] for j in range(len(o1))]
        # remove all padding tokens from generated output for cycle detection
        o1_cpu = [o[o != tokenizer.pad_token_id] for o in o1_cpu]
        reps = []
        for dbg_idx, o in enumerate(tqdm(o1_cpu, desc=f"cycle_detect batch {i}", leave=False)):
            rep = detect_cycles(o[plengths[dbg_idx]:], return_index=True, pad_token_id=tokenizer.pad_token_id)
            reps.append(rep)
        # 4. Build natural and ICL inputs
        natural_input = [o1_cpu[j][:plengths[j] + rep[3]].tolist() + rep[0]*n_cycles for j, rep in enumerate(reps) if rep[0] is not None]
        icl_input = [rep[0]*n_cycles for rep in reps if rep[0] is not None]
        expected_next = [rep[0][0] for rep in reps if rep[0] is not None]
        
        data_index.extend([i + j for j in range(len(reps)) if reps[j][0] is not None])
        rep_index.extend([reps[j][3] for j in range(len(reps)) if reps[j][0] is not None])
        total_natural += len(natural_input)
        total_icl += len(icl_input)

        # no repetition ICL
        # find sequences with no cycles
        no_cycle_icl_input = [o1_cpu[j][:plengths[j]].tolist()*n_cycles for j, rep in enumerate(reps) if rep[0] is None]
        total_no_cycle += len(no_cycle_icl_input)
        no_cycle_index.extend([i + j for j in range(len(reps)) if reps[j][0] is None])  # Store sample indices, not prompt lengths
        expected_next_no_cycle = [o1_cpu[j][0] for j, rep in enumerate(reps) if rep[0] is None]
        # ===== NATURAL INPUT PROCESSING =====
        if natural_input and not no_head_analysis:
            # We already ran the model on the full batch, now filter outputs for sequences with cycles
            # Create mapping from original batch indices to cycle indices
            cycle_indices = [j for j, rep in enumerate(reps) if rep[0] is not None]
            
            if len(cycle_indices) > 0:
                # Extract cycle information for sequences with cycles
                cycle_start_positions = [reps[j][3] for j in cycle_indices]  # rep[3] is cycle_start_index
                original_lengths = [plengths[j] for j in cycle_indices]  # Original prompt lengths
                
                # Filter final layer logits to only include sequences with cycles
                if hooked_model.has_final_logits():
                    all_logits = hooked_model.get_all_logits()  # Get ALL logits, not just final
                    filtered_logits = all_logits[cycle_indices]
                    # Temporarily replace the final logits with filtered version
                    # original_logits = hooked_model.final_layer_logits
                    hooked_model.final_layer_logits = filtered_logits
                
                # Filter attention outputs to only include sequences with cycles
                filtered_attn_outputs = []
                for layer_name, layer_data in hooked_model.attn_outputs:
                    if isinstance(layer_data, (list, tuple)):
                        # Handle list of tensors
                        filtered_layer_data = [tensor[cycle_indices] if hasattr(tensor, '__getitem__') else tensor for tensor in layer_data]
                    elif hasattr(layer_data, 'shape') and layer_data.shape[0] >= len(cycle_indices):
                        # Handle single tensor - filter batch dimension
                        filtered_layer_data = layer_data[cycle_indices]
                    else:
                        filtered_layer_data = layer_data
                    filtered_attn_outputs.append((layer_name, filtered_layer_data))
                
                # Temporarily replace attention outputs with filtered version
                original_attn_outputs = hooked_model.attn_outputs
                hooked_model.attn_outputs = filtered_attn_outputs
                
                # Compute contrasts directly using filtered outputs and cycle positions
                natural_acts, natural_next_highest, n_success_nat = compute_direct_contrasts(
                    hooked_model, expected_next, lens, device, f"{i}_natural",
                    cycle_start_positions=cycle_start_positions,
                    original_sequence_lengths=original_lengths
                )
                total_successful_nat += n_success_nat
                
                # Restore original outputs
                # hooked_model.final_layer_logits = original_logits
                # hooked_model.attn_outputs = original_attn_outputs
                
                # Accumulate results
                for k in natural_acts:
                    acts_cyc.setdefault(k, []).extend(natural_acts[k])
                for k in natural_next_highest:
                    next_highest.setdefault(k, []).extend(natural_next_highest[k])
        
        # ===== ICL INPUT PROCESSING =====
        # Check if ICL input has meaningful content (not just empty sequences)
        has_meaningful_icl = icl_input and any(len(seq) > 0 for seq in icl_input)
        
        if not has_meaningful_icl:
            print(f"Batch {i}: Skipping ICL processing - no meaningful ICL sequences (n_cycles=0 or no cycles detected)")
            continue
        
        if has_meaningful_icl and not no_head_analysis:
            # For ICL, we need to run the model on just the cycles (different input)
            # Helper for padding and batching
            def pad_and_batch(seqs):
                if not seqs:
                    return None, None
                max_len = max(len(seq) for seq in seqs)
                input_ids = [([tokenizer.pad_token_id] * (max_len - len(seq)) + seq) for seq in seqs]
                attention_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
                return torch.tensor(input_ids, device=device), torch.tensor(attention_mask, device=device)
            
            # Prepare ICL input for model - filter out empty sequences
            non_empty_icl_input = [seq for seq in icl_input if len(seq) > 0]
            if not non_empty_icl_input:
                warn("No non-empty ICL inputs to process")
                continue
            icl_input_ids_tensor, icl_attention_mask_tensor = pad_and_batch(non_empty_icl_input)
            icl_input_dict = {'input_ids': icl_input_ids_tensor, 'attention_mask': icl_attention_mask_tensor}
            
            # Filter expected_next to match the non-empty ICL sequences
            # Create indices of non-empty sequences
            non_empty_indices = [idx for idx, seq in enumerate(icl_input) if len(seq) > 0]
            expected_next_filtered = [expected_next[idx] for idx in non_empty_indices if idx < len(expected_next)]
            
            # Run model on ICL input to capture outputs
            hooked_model.clear()
            with torch.no_grad():
                _ = hooked_model.generate(**icl_input_dict, do_sample=False, max_new_tokens=1)
            
            # Compute contrasts directly for ICL
            icl_acts_batch, icl_next_highest_batch, n_success_icl = compute_direct_contrasts(
                hooked_model, expected_next_filtered, lens, device, f"{i}_icl"
            )
            total_successful_icl += n_success_icl
            
            # Accumulate ICL results  
            for k in icl_acts_batch:
                icl_acts.setdefault(k, []).extend(icl_acts_batch[k])
            for k in icl_next_highest_batch:
                icl_next_highest.setdefault(k, []).extend(icl_next_highest_batch[k])

        ## No Cycle ICL Processing
        if no_cycle_icl_input and not no_head_analysis:
            # Prepare no-cycle ICL input for model
            no_cycle_icl_input_ids_tensor, no_cycle_icl_attention_mask_tensor = pad_and_batch(no_cycle_icl_input)
            no_cycle_icl_input_dict = {'input_ids': no_cycle_icl_input_ids_tensor, 'attention_mask': no_cycle_icl_attention_mask_tensor}
            
            # Run model on no-cycle ICL input to capture outputs
            hooked_model.clear()
            with torch.no_grad():
                _ = hooked_model.generate(**no_cycle_icl_input_dict, do_sample=False, max_new_tokens=1)
            
            # Compute contrasts directly for no-cycle ICL
            no_cycle_icl_acts_batch, no_cycle_icl_next_highest_batch, n_success_no_cycle = compute_direct_contrasts(
                hooked_model, expected_next_no_cycle, lens, device, f"{i}_no_cycle_icl"
            )
            total_successful_no_cycle += n_success_no_cycle
            
            # Accumulate no-cycle ICL results  
            for k in no_cycle_icl_acts_batch:
                no_cycle_icl_acts.setdefault(k, []).extend(no_cycle_icl_acts_batch[k])
            for k in no_cycle_icl_next_highest_batch:
                no_cycle_icl_next_highest.setdefault(k, []).extend(no_cycle_icl_next_highest_batch[k])

        # Clear model outputs for next batch
        hooked_model.clear()
    
    # Compute heatmaps with both means and variances
    def build_heatmap_with_variance(acts, next_highest):
        if acts and list(acts.keys()):
            first_key = list(acts.keys())[0]
            if acts[first_key]:
                # For single layer analysis, create arrays for means and variances
                means = np.zeros(len(acts))  # Shape: (num_layers,)
                variances = np.zeros(len(acts))  # Shape: (num_layers,)
                
                for i, (k, v) in enumerate(acts.items()):
                    # Concatenate all arrays first, then compute statistics
                    if v and len(v) > 0:
                        try:                            
                            # For single layer analysis, concatenate all samples and compute statistics
                            v_concat = np.concatenate(v, axis=0)  # Combine all samples from all batches
                            c_acts_mean = np.mean(v_concat)  # Single scalar mean across all samples
                            c_acts_var = np.var(v_concat)   # Single scalar variance across all samples
                            
                        except ValueError as e:
                            print(f"Error in heatmap building for {k}: {e}")
                            # Fallback: flatten everything and compute statistics
                            v_flat = np.concatenate([vi.flatten() for vi in v])
                            c_acts_mean = np.mean(v_flat)
                            c_acts_var = np.var(v_flat)
                    else:
                        c_acts_mean = 0.0  # Scalar fallback for empty v
                        c_acts_var = 0.0
                    
                    # Same fix for next_highest
                    if next_highest[k] and len(next_highest[k]) > 0:
                        try:
                            # For single layer analysis, concatenate all samples and compute statistics
                            next_concat = np.concatenate(next_highest[k], axis=0)  # Combine all samples
                            next_acts_mean = np.mean(next_concat)  # Single scalar mean
                            next_acts_var = np.var(next_concat)    # Single scalar variance
                        except ValueError as e:
                            print(f"Error in next_highest heatmap building for {k}: {e}")
                            # Fallback: flatten everything and compute statistics
                            next_flat = np.concatenate([ni.flatten() for ni in next_highest[k]])
                            next_acts_mean = np.mean(next_flat)
                            next_acts_var = np.var(next_flat)
                    else:
                        next_acts_mean = 0.0  # Scalar fallback for empty next_highest[k]
                        next_acts_var = 0.0
                    
                    # Compute contrast mean and variance
                    contrast_mean = c_acts_mean - next_acts_mean
                    # For variance of difference: Var(X - Y) = Var(X) + Var(Y) - 2*Cov(X,Y)
                    # If we assume independence, Cov(X,Y) = 0, so Var(X - Y) = Var(X) + Var(Y)
                    contrast_var = c_acts_var + next_acts_var
                    
                    means[i] = contrast_mean
                    variances[i] = contrast_var
                
                return means, variances
        return np.zeros((0,)), np.zeros((0,))
    
    # Build heatmaps with variances
    heatmap_mean, heatmap_var = build_heatmap_with_variance(acts_cyc, next_highest)
    
    if icl_acts:
        icl_heatmap_mean, icl_heatmap_var = build_heatmap_with_variance(icl_acts, icl_next_highest)
    else:
        icl_heatmap_mean, icl_heatmap_var = None, None
    
    if no_cycle_icl_acts:
        no_cycle_icl_heatmap_mean, no_cycle_icl_heatmap_var = build_heatmap_with_variance(no_cycle_icl_acts, no_cycle_icl_next_highest)
    else:
        no_cycle_icl_heatmap_mean, no_cycle_icl_heatmap_var = None, None
    
    print(f"Final results:")
    print(f"- Natural: {len(acts_cyc)} attention heads processed")
    print(f"- ICL: {len(icl_acts)} attention heads processed")
    print(f"- No-cycle ICL: {len(no_cycle_icl_acts)} attention heads processed")
    print(f"- Total natural samples: {total_natural}")
    print(f"- Total ICL samples: {total_icl}")
    print(f"- Total no-cycle ICL samples: {total_no_cycle}")
    print(f"- Total successful natural predictions: {total_successful_nat}")
    print(f"- Total successful ICL predictions: {total_successful_icl}")
    print(f"- Total successful no-cycle ICL predictions: {total_successful_no_cycle}")
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    # Return results in the expected format (indexed by layer) with both means and variances
    if layers is not None and len(layers) == 1:
        # Single layer case
        layer_idx = layers[0]
        heatmap_dict = {layer_idx: {'mean': heatmap_mean, 'var': heatmap_var}}
        icl_heatmap_dict = {layer_idx: {'mean': icl_heatmap_mean, 'var': icl_heatmap_var}} if icl_heatmap_mean is not None else {layer_idx: None}
        no_cycle_icl_heatmap_dict = {layer_idx: {'mean': no_cycle_icl_heatmap_mean, 'var': no_cycle_icl_heatmap_var}} if no_cycle_icl_heatmap_mean is not None else {layer_idx: None}
    else:
        # Multiple layers case (though this function seems to handle only one layer at a time)
        heatmap_dict = {0: {'mean': heatmap_mean, 'var': heatmap_var}}  # Default to layer 0
        icl_heatmap_dict = {0: {'mean': icl_heatmap_mean, 'var': icl_heatmap_var}} if icl_heatmap_mean is not None else {0: None}
        no_cycle_icl_heatmap_dict = {0: {'mean': no_cycle_icl_heatmap_mean, 'var': no_cycle_icl_heatmap_var}} if no_cycle_icl_heatmap_mean is not None else {0: None}

    return heatmap_dict, total_natural, icl_heatmap_dict, total_icl, no_cycle_icl_heatmap_dict, total_no_cycle, data_index, rep_index, no_cycle_index
