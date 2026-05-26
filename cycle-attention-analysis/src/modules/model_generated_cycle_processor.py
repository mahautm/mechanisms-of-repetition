import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/mmahaut/projects/parrots')
from parrots.cycle_detection import detect_cycles

class ModelGeneratedCycleProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Set padding side for decoder-only models
        self.original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def process_texts(self, texts, model, n_cycles, max_length, max_new_tokens, batch_size):
        """
        Process texts by generating with the model first, then detecting cycles in the generation.
        Creates four types of sequences:
        1. Natural: Model-generated cycles extended naturally
        2. ICL: Detected cycles repeated artificially 
        3. No-cycle: Sequences without any cycles
        4. No-cycle ICL: Non-repetitive patterns repeated artificially (NEW!)
        """
        
        natural_sequences = []
        icl_sequences = []
        no_cycle_sequences = []
        no_cycle_icl_sequences = []  # NEW!
        
        device = model.device
        
        # Process texts in batches        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Step 1: Generate with model (like contrast_analysis does)
            try:
                generated_sequences, prompt_lengths = self._generate_from_texts(
                    batch_texts, model, max_length, max_new_tokens, device
                )
                
                # Step 2: Detect cycles in generated output
                cycle_results = self._detect_cycles_in_generation(
                    generated_sequences, prompt_lengths
                )
                
                # Step 3: Create different sequence types based on detected cycles
                batch_natural, batch_icl, batch_no_cycle, batch_no_cycle_icl = self._create_sequence_types(
                    cycle_results, generated_sequences, prompt_lengths, 
                    batch_texts, model, n_cycles, max_new_tokens
                )
                
                natural_sequences.extend(batch_natural)
                icl_sequences.extend(batch_icl)
                no_cycle_sequences.extend(batch_no_cycle)
                no_cycle_icl_sequences.extend(batch_no_cycle_icl)  # NEW!
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        return natural_sequences, icl_sequences, no_cycle_sequences, no_cycle_icl_sequences
    
    def _generate_from_texts(self, batch_texts, model, max_length, max_new_tokens, device):
        """Generate sequences from input texts using the model."""
        
        # Tokenize input texts
        pretokenized = [
            self.tokenizer(text, return_tensors="pt", padding=True, 
                          truncation=True, max_length=max_length) 
            for text in batch_texts
        ]
        
        # Pad sequences to same length
        input_ids_list = [b['input_ids'].squeeze(0) for b in pretokenized]
        attention_mask_list = [b['attention_mask'].squeeze(0) for b in pretokenized]
        
        from torch.nn.utils.rnn import pad_sequence
        input_ids = pad_sequence(input_ids_list, batch_first=True, 
                                padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        toked = {
            'input_ids': input_ids.to(device), 
            'attention_mask': attention_mask.to(device)
        }
        
        # Generate sequences
        with torch.no_grad():
            generated = model.generate(
                **toked, 
                do_sample=False, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Get prompt lengths (number of non-padding tokens in input)
        prompt_lengths = toked["attention_mask"].sum(dim=1).tolist()
        
        # Convert to CPU and remove padding tokens
        generated_cpu = []
        for seq in generated:
            seq_cpu = seq.detach().cpu()
            seq_no_pad = seq_cpu[seq_cpu != self.tokenizer.pad_token_id]
            generated_cpu.append(seq_no_pad)
        
        return generated_cpu, prompt_lengths
    
    def _detect_cycles_in_generation(self, generated_sequences, prompt_lengths):
        """Detect cycles in the generated portions of sequences."""
        
        cycle_results = []
        
        for seq_idx, (seq, prompt_len) in enumerate(zip(generated_sequences, prompt_lengths)):
            # Extract only the generated portion (after the prompt)
            generated_portion = seq[prompt_len:]
            
            if len(generated_portion) < 6:  # Need minimum length for cycle detection
                cycle_results.append({
                    'seq_idx': seq_idx,
                    'has_cycle': False,
                    'cycle_data': None,
                    'full_sequence': seq.tolist(),
                    'prompt_length': prompt_len,
                    'generated_length': len(generated_portion),
                    'generated_portion': generated_portion.tolist()
                })
                continue
            
            # Use the same cycle detection as contrast_analysis
            try:
                cycle_data = detect_cycles(
                    generated_portion, 
                    return_index=True, 
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # cycle_data format: [cycle_tokens, cycle_length, n_repeats, cycle_start_index]
                has_cycle = cycle_data[0] is not None
                
                cycle_results.append({
                    'seq_idx': seq_idx,
                    'has_cycle': has_cycle,
                    'cycle_data': cycle_data,
                    'full_sequence': seq.tolist(),
                    'prompt_length': prompt_len,
                    'generated_length': len(generated_portion),
                    'generated_portion': generated_portion.tolist()
                })
                
            except Exception as e:
                print(f"Error detecting cycles in sequence {seq_idx}: {e}")
                cycle_results.append({
                    'seq_idx': seq_idx,
                    'has_cycle': False,
                    'cycle_data': None,
                    'full_sequence': seq.tolist(),
                    'prompt_length': prompt_len,
                    'generated_length': len(generated_portion),
                    'generated_portion': generated_portion.tolist()
                })
        
        return cycle_results
    
    def _create_sequence_types(self, cycle_results, generated_sequences, prompt_lengths, 
                              original_texts, model, n_cycles, max_new_tokens):
        """Create natural, ICL, no-cycle, and no-cycle ICL sequences."""
        
        natural_sequences = []
        icl_sequences = []
        no_cycle_sequences = []
        no_cycle_icl_sequences = []  # NEW!
        
        for result in cycle_results:
            seq_idx = result['seq_idx']
            original_text = original_texts[seq_idx]
            
            if result['has_cycle']:
                # Create natural sequence (original prompt + detected cycle extended)
                cycle_tokens, cycle_length, n_repeats, cycle_start_idx = result['cycle_data']
                
                natural_seq_data = {
                    'sequence': result['full_sequence'],
                    'original_text': original_text,
                    'cycle': cycle_tokens,
                    'cycle_text': self.tokenizer.decode(cycle_tokens),
                    'cycle_start_idx': cycle_start_idx,  # Relative to generated portion
                    'n_cycles': n_repeats,
                    'prompt_length': result['prompt_length'],
                    'generated_length': result['generated_length'],
                    'sequence_type': 'natural_generated'
                }
                natural_sequences.append(natural_seq_data)
                
                # Create ICL sequence (just the cycle repeated)
                icl_sequence_tokens = cycle_tokens * n_cycles
                
                # Generate continuation for ICL sequence
                try:
                    icl_continuation = self._generate_icl_continuation(
                        icl_sequence_tokens, model, max_new_tokens
                    )
                    
                    icl_seq_data = {
                        'sequence': icl_continuation,
                        'original_text': original_text,
                        'cycle': cycle_tokens,
                        'cycle_text': self.tokenizer.decode(cycle_tokens),
                        'cycle_start_idx': 0,
                        'n_cycles': n_cycles,
                        'prompt_length': len(icl_sequence_tokens),
                        'generated_length': len(icl_continuation) - len(icl_sequence_tokens),
                        'sequence_type': 'icl_generated'
                    }
                    icl_sequences.append(icl_seq_data)
                    
                except Exception as e:
                    print(f"Error creating ICL sequence for {seq_idx}: {e}")
            
            else:
                # No cycle detected - create no-cycle sequence AND no-cycle ICL
                no_cycle_seq_data = {
                    'sequence': result['full_sequence'],
                    'original_text': original_text,
                    'prompt_length': result['prompt_length'],
                    'generated_length': result['generated_length'],
                    'sequence_type': 'no_cycle_generated'
                }
                no_cycle_sequences.append(no_cycle_seq_data)
                
                # NEW: Create no-cycle ICL sequence
                # Take a chunk from the generated portion and repeat it artificially
                generated_portion = result['generated_portion']
                
                if len(generated_portion) >= 4:  # Need minimum length to create a pattern
                    no_cycle_icl_data = self._create_no_cycle_icl_sequence(
                        generated_portion, original_text, model, n_cycles, max_new_tokens
                    )
                    if no_cycle_icl_data:
                        no_cycle_icl_sequences.append(no_cycle_icl_data)
        
        return natural_sequences, icl_sequences, no_cycle_sequences, no_cycle_icl_sequences
    
    def _create_no_cycle_icl_sequence(self, generated_portion, original_text, model, n_cycles, max_new_tokens):
        """
        Create a no-cycle ICL sequence by taking a non-repetitive part and repeating it.
        This is the control condition: what happens when we force repetition on something
        that the model never naturally repeated?
        """
        
        try:
            # Take a chunk from the middle of the generated portion
            # (avoiding start/end artifacts)
            gen_len = len(generated_portion)
            chunk_size = min(max(3, gen_len // 4), 8)  # 3-8 tokens
            
            # Start from middle-ish position
            start_pos = max(1, gen_len // 3)
            end_pos = min(start_pos + chunk_size, gen_len)
            
            if end_pos <= start_pos:
                return None
            
            # Extract the chunk that was NOT repetitive
            no_cycle_chunk = generated_portion[start_pos:end_pos]
            
            # Repeat it artificially
            repeated_chunk = no_cycle_chunk * n_cycles
            
            # Generate continuation
            input_ids = torch.tensor([repeated_chunk], device=model.device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            full_sequence = generated[0].cpu().tolist()
            
            return {
                'sequence': full_sequence,
                'original_text': original_text,
                'cycle': no_cycle_chunk,
                'cycle_text': self.tokenizer.decode(no_cycle_chunk),
                'cycle_start_idx': 0,
                'n_cycles': n_cycles,
                'prompt_length': len(repeated_chunk),
                'generated_length': len(full_sequence) - len(repeated_chunk),
                'sequence_type': 'no_cycle_icl',
                'note': 'Artificially repeated non-repetitive pattern'
            }
            
        except Exception as e:
            print(f"Error creating no-cycle ICL sequence: {e}")
            return None
    
    def _generate_icl_continuation(self, icl_tokens, model, max_new_tokens):
        """Generate continuation for ICL sequence."""
        
        input_ids = torch.tensor([icl_tokens], device=model.device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return generated[0].cpu().tolist()
    
    def __del__(self):
        """Restore original tokenizer settings."""
        if hasattr(self, 'original_padding_side'):
            self.tokenizer.padding_side = self.original_padding_side