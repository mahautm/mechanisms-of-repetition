import torch
import numpy as np
from tqdm import tqdm
import re
from collections import Counter

class CycleProcessorFixed:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process_texts(self, texts, model, n_cycles, max_length, max_new_tokens, batch_size):
        """Process texts to identify different types of sequences with cycles."""
        
        natural_sequences = []
        icl_sequences = []
        no_cycle_sequences = []
        
        # Process texts in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Process each text in the batch
            for text in batch_texts:
                try:
                    # Detect cycles in the original text
                    cycles = self._detect_cycles(text, min_length=3, max_length=50)
                    
                    if cycles:
                        # Found cycles in the text - create natural sequence
                        sequence_data = self._create_natural_sequence(
                            text, cycles, model, max_length, max_new_tokens
                        )
                        if sequence_data:
                            natural_sequences.append(sequence_data)
                    else:
                        # No cycles found - create ICL sequence or no-cycle sequence
                        icl_data, no_cycle_data = self._create_icl_and_no_cycle_sequences(
                            text, model, n_cycles, max_length, max_new_tokens
                        )
                        
                        if icl_data:
                            icl_sequences.append(icl_data)
                        if no_cycle_data:
                            no_cycle_sequences.append(no_cycle_data)
                
                except Exception as e:
                    print(f"Error processing text: {e}")
                    continue
        
        return natural_sequences, icl_sequences, no_cycle_sequences
    
    def _detect_cycles(self, text, min_length=3, max_length=50):
        """Detect repeating patterns (cycles) in text."""
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        cycles = []
        
        # Look for repeating token sequences
        for cycle_len in range(min_length, min(max_length, len(tokens) // 2)):
            for start_pos in range(len(tokens) - cycle_len * 2):
                
                # Extract potential cycle
                cycle_tokens = tokens[start_pos:start_pos + cycle_len]
                
                # Check if this pattern repeats
                repeats = 1
                pos = start_pos + cycle_len
                
                while pos + cycle_len <= len(tokens):
                    next_segment = tokens[pos:pos + cycle_len]
                    if next_segment == cycle_tokens:
                        repeats += 1
                        pos += cycle_len
                    else:
                        break
                
                # If we found 2 or more repeats, it's a cycle
                if repeats >= 2:
                    cycle_text = self.tokenizer.decode(cycle_tokens)
                    cycles.append({
                        'tokens': cycle_tokens,
                        'text': cycle_text,
                        'start_pos': start_pos,
                        'length': cycle_len,
                        'repeats': repeats
                    })
        
        # Remove overlapping cycles, keep the longest ones
        cycles = self._remove_overlapping_cycles(cycles)
        
        return cycles
    
    def _remove_overlapping_cycles(self, cycles):
        """Remove overlapping cycles, preferring longer ones."""
        
        if not cycles:
            return cycles
        
        # Sort by length (descending) and then by number of repeats
        cycles.sort(key=lambda x: (x['length'], x['repeats']), reverse=True)
        
        non_overlapping = []
        used_positions = set()
        
        for cycle in cycles:
            start = cycle['start_pos']
            end = start + cycle['length'] * cycle['repeats']
            
            # Check if this cycle overlaps with any already selected
            cycle_positions = set(range(start, end))
            if not cycle_positions.intersection(used_positions):
                non_overlapping.append(cycle)
                used_positions.update(cycle_positions)
        
        return non_overlapping
    
    def _create_natural_sequence(self, text, cycles, model, max_length, max_new_tokens):
        """Create a natural sequence from text that already contains cycles."""
        
        if not cycles:
            return None
        
        # Use the longest cycle
        best_cycle = max(cycles, key=lambda x: x['length'] * x['repeats'])
        
        # Tokenize the full text
        full_tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[:max_length]
        
        # Generate continuation to see if model continues the cycle
        try:
            input_ids = torch.tensor([full_tokens]).to(model.device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_tokens = generated[0].cpu().tolist()
            
            return {
                'sequence': generated_tokens,
                'original_text': text,
                'cycle': best_cycle['tokens'],
                'cycle_text': best_cycle['text'],
                'cycle_start_idx': best_cycle['start_pos'],
                'n_cycles': best_cycle['repeats'],
                'prompt_length': len(full_tokens),
                'generated_length': len(generated_tokens) - len(full_tokens)
            }
            
        except Exception as e:
            print(f"Error generating natural sequence: {e}")
            return None
    
    def _create_icl_and_no_cycle_sequences(self, text, model, n_cycles, max_length, max_new_tokens):
        """Create ICL sequence by forcing repetition, and a no-cycle control."""
        
        # Create a simple repetitive pattern for ICL
        base_tokens = self.tokenizer.encode(text)[:20]  # Take first 20 tokens
        
        if len(base_tokens) < 3:
            return None, None
        
        # Create ICL sequence by repeating the pattern
        icl_tokens = base_tokens * n_cycles
        
        # Truncate if too long
        if len(icl_tokens) > max_length - max_new_tokens:
            icl_tokens = icl_tokens[:max_length - max_new_tokens]
        
        # Generate continuation
        try:
            # ICL sequence
            input_ids = torch.tensor([icl_tokens]).to(model.device)
            
            with torch.no_grad():
                icl_generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            icl_result = icl_generated[0].cpu().tolist()
            
            icl_data = {
                'sequence': icl_result,
                'original_text': text,
                'cycle': base_tokens,
                'cycle_text': self.tokenizer.decode(base_tokens),
                'cycle_start_idx': 0,
                'n_cycles': n_cycles,
                'prompt_length': len(icl_tokens),
                'generated_length': len(icl_result) - len(icl_tokens)
            }
            
            # No-cycle sequence (just the original text without forced repetition)
            original_tokens = self.tokenizer.encode(text)
            if len(original_tokens) > max_length - max_new_tokens:
                original_tokens = original_tokens[:max_length - max_new_tokens]
            
            input_ids = torch.tensor([original_tokens]).to(model.device)
            
            with torch.no_grad():
                no_cycle_generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            no_cycle_result = no_cycle_generated[0].cpu().tolist()
            
            no_cycle_data = {
                'sequence': no_cycle_result,
                'original_text': text,
                'prompt_length': len(original_tokens),
                'generated_length': len(no_cycle_result) - len(original_tokens)
            }
            
            return icl_data, no_cycle_data
            
        except Exception as e:
            print(f"Error creating ICL/no-cycle sequences: {e}")
            return None, None