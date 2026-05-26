#!/usr/bin/env python3
"""
Newline Token Verification and Attention Analysis

This script verifies we're identifying and removing the correct newline token,
and analyzes what attention heads focus on during newline processing.
"""

import torch
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/parrots/aa_fortu/modules')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src/modules')

try:
    from model_utils import load_model_and_tokenizer
    from parrots.cycle_detection import detect_cycles
    print("✅ Imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def verify_newline_token_identification():
    """Verify we're identifying the correct newline token."""
    print("🔍 VERIFYING NEWLINE TOKEN IDENTIFICATION")
    print("=" * 50)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    
    # Test different newline representations
    newline_tests = [
        "\\n",       # The literal characters \n  
        "\n",        # Actual newline character
        "\\r\\n",    # Windows line ending as string
        "\r\n",      # Actual Windows line ending
        "\t",        # Tab character
        " ",         # Space character
    ]
    
    print("Token ID analysis:")
    for i, test_char in enumerate(newline_tests):
        try:
            token_ids = tokenizer.encode(test_char, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids)
            print(f"{i+1}. {repr(test_char):>8} → Token IDs: {token_ids} → Decoded: {repr(decoded)}")
        except Exception as e:
            print(f"{i+1}. {repr(test_char):>8} → Error: {e}")
    
    # Get the newline token ID used in experiments
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    print(f"\n✅ Experiment uses newline token ID: {newline_token_id}")
    
    # Test tokenization of text with newlines
    test_text = "Line 1\nLine 2\nLine 3"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    
    print(f"\nTest text: {repr(test_text)}")
    print("Token breakdown:")
    for i, (token_id, decoded) in enumerate(zip(tokens, decoded_tokens)):
        is_newline = "← NEWLINE" if token_id == newline_token_id else ""
        print(f"  {i:2d}: {token_id:5d} → {repr(decoded):>10} {is_newline}")
    
    return newline_token_id, tokenizer


def analyze_attention_to_newlines(model, tokenizer, newline_token_id):
    """Analyze what attention heads focus on during newline processing."""
    print("\n🔍 ANALYZING ATTENTION TO NEWLINES")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Test text with strategic newline placement
    test_texts = [
        "First paragraph here.\nSecond paragraph continues.\nThird paragraph follows.",
        "Step 1: Initialize\nStep 2: Process\nStep 3: Complete",
        "Category A items\nCategory B items\nCategory C items",
        "Beginning.\n\nMiddle section.\n\nEnd section."  # Double newlines
    ]
    
    results = {}
    
    for text_idx, text in enumerate(test_texts):
        print(f"\nAnalyzing text {text_idx + 1}: {repr(text[:50])}")
        
        # Tokenize
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        tokens = [tokenizer.decode([t]) for t in input_ids[0]]
        
        # Find newline positions
        newline_positions = [i for i, token_id in enumerate(input_ids[0]) 
                           if token_id == newline_token_id]
        
        print(f"  Newline positions: {newline_positions}")
        print(f"  Token sequence: {[repr(t) for t in tokens]}")
        
        if not newline_positions:
            print("  ⚠️ No newlines found!")
            continue
        
        # Get attention weights
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs.attentions  # tuple of attention tensors
        
        text_results = {
            'text': text,
            'tokens': tokens,
            'newline_positions': newline_positions,
            'attention_analysis': {}
        }
        
        # Analyze attention patterns for each layer
        for layer_idx in [15, 17, 19, 21, 23]:  # Focus on later layers
            if layer_idx < len(attentions):
                attention = attentions[layer_idx]  # [1, num_heads, seq_len, seq_len]
                
                layer_analysis = analyze_layer_attention_to_newlines(
                    attention, newline_positions, tokens, layer_idx
                )
                text_results['attention_analysis'][f'layer_{layer_idx}'] = layer_analysis
        
        results[f'text_{text_idx}'] = text_results
    
    return results


def analyze_layer_attention_to_newlines(attention, newline_positions, tokens, layer_idx):
    """Analyze a single layer's attention to newlines."""
    # attention: [1, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attention.shape
    
    layer_results = {
        'layer': layer_idx,
        'num_heads': num_heads,
        'newline_attention_patterns': []
    }
    
    for newline_pos in newline_positions:
        if newline_pos >= seq_len:
            continue
            
        newline_pattern = {
            'newline_position': newline_pos,
            'context': tokens[max(0, newline_pos-2):newline_pos+3],
            'head_analysis': []
        }
        
        # Analyze each attention head
        for head_idx in range(num_heads):
            head_attn = attention[0, head_idx]  # [seq_len, seq_len]
            
            # What does the newline attend to?
            newline_attention_out = head_attn[newline_pos]  # What newline attends to
            top_attended_indices = torch.topk(newline_attention_out, k=min(5, seq_len)).indices
            
            # What attends to the newline?
            newline_attention_in = head_attn[:, newline_pos]  # What attends to newline
            top_attending_indices = torch.topk(newline_attention_in, k=min(5, seq_len)).indices
            
            head_analysis = {
                'head': head_idx,
                'newline_attends_to': [
                    {
                        'position': int(idx),
                        'token': tokens[idx] if idx < len(tokens) else '<PAD>',
                        'weight': float(newline_attention_out[idx])
                    }
                    for idx in top_attended_indices
                ],
                'attends_to_newline': [
                    {
                        'position': int(idx),
                        'token': tokens[idx] if idx < len(tokens) else '<PAD>',
                        'weight': float(newline_attention_in[idx])
                    }
                    for idx in top_attending_indices
                ],
                'self_attention_weight': float(head_attn[newline_pos, newline_pos])
            }
            
            newline_pattern['head_analysis'].append(head_analysis)
        
        layer_results['newline_attention_patterns'].append(newline_pattern)
    
    return layer_results


def test_newline_removal_effectiveness():
    """Test whether newline removal actually prevents repetition."""
    print("\n🔍 TESTING NEWLINE REMOVAL EFFECTIVENESS")
    print("=" * 50)
    
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Test prompts designed to potentially trigger repetition
    test_prompts = [
        "The main steps are:\n1. First step\n2. Second step\n3.",
        "Categories include:\nType A items\nType B items\nType C",
        "Process overview:\nInitialize system\nRun analysis\nGenerate report",
        "Key points:\n- Point one\n- Point two\n- Point three"
    ]
    
    results = []
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: {repr(prompt)}")
        
        # Generate with original (with newlines)
        input_ids_orig = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_orig = model.generate(
                input_ids_orig,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)
        
        # Generate with newlines removed (replaced with spaces)
        prompt_no_newlines = prompt.replace('\n', ' ')
        input_ids_no_nl = tokenizer.encode(prompt_no_newlines, return_tensors="pt").to(device)
        with torch.no_grad():
            output_no_nl = model.generate(
                input_ids_no_nl,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_no_nl = tokenizer.decode(output_no_nl[0], skip_special_tokens=True)
        
        # Extract only generated parts
        orig_generated = generated_orig[len(prompt):].strip()
        no_nl_generated = generated_no_nl[len(prompt_no_newlines):].strip()
        
        # Analyze repetition using standard detect_cycles
        orig_tokens = tokenizer(orig_generated, return_tensors='pt')['input_ids'][0].tolist()
        no_nl_tokens = tokenizer(no_nl_generated, return_tensors='pt')['input_ids'][0].tolist()
        
        orig_cycle, orig_size, orig_count = detect_cycles(orig_tokens)
        no_nl_cycle, no_nl_size, no_nl_count = detect_cycles(no_nl_tokens)
        
        result = {
            'prompt': prompt,
            'with_newlines': {
                'generated': orig_generated[:200],
                'cycle_count': orig_count if orig_count else 0,
                'cycle_size': orig_size,
            },
            'without_newlines': {
                'generated': no_nl_generated[:200],
                'cycle_count': no_nl_count if no_nl_count else 0,
                'cycle_size': no_nl_size,
            },
            'newline_effect': {
                'repetition_reduced': (orig_count if orig_count else 0) > (no_nl_count if no_nl_count else 0),
                'cycle_count_change': (orig_count if orig_count else 0) - (no_nl_count if no_nl_count else 0)
            }
        }
        
        results.append(result)
        
        print(f"  With newlines: {orig_count if orig_count else 0} cycles")
        print(f"  Without newlines: {no_nl_count if no_nl_count else 0} cycles")
        print(f"  Repetition reduced: {result['newline_effect']['repetition_reduced']}")
    
    return results


def main():
    """Run complete newline token verification and attention analysis."""
    print("🚀 NEWLINE TOKEN VERIFICATION AND ATTENTION ANALYSIS")
    print("=" * 60)
    
    # 1. Verify token identification
    newline_token_id, tokenizer = verify_newline_token_identification()
    
    # 2. Load model for attention analysis
    model, _ = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    
    # 3. Analyze attention patterns
    attention_results = analyze_attention_to_newlines(model, tokenizer, newline_token_id)
    
    # 4. Test removal effectiveness
    removal_results = test_newline_removal_effectiveness()
    
    # 5. Summary analysis
    print("\n📊 SUMMARY ANALYSIS")
    print("=" * 50)
    
    print(f"✅ Newline token ID verified: {newline_token_id}")
    print(f"✅ Analyzed attention in {len(attention_results)} test cases")
    print(f"✅ Tested removal effectiveness on {len(removal_results)} prompts")
    
    # Attention pattern summary
    print("\n🔍 Key Attention Findings:")
    for text_key, text_data in attention_results.items():
        if text_data['newline_positions']:
            print(f"  {text_key}: Found newlines at positions {text_data['newline_positions']}")
    
    # Removal effectiveness summary
    print("\n🔧 Newline Removal Effectiveness:")
    reduction_count = sum(1 for r in removal_results if r['newline_effect']['repetition_reduced'])
    print(f"  Repetition reduced in {reduction_count}/{len(removal_results)} cases")
    
    total_cycle_reduction = sum(r['newline_effect']['cycle_count_change'] for r in removal_results)
    print(f"  Total cycle count reduction: {total_cycle_reduction}")
    
    if total_cycle_reduction > 0:
        print("  ✅ Newline removal shows positive effect on repetition reduction")
    else:
        print("  ⚠️ Newline removal shows minimal effect on repetition")
    
    return {
        'token_verification': newline_token_id,
        'attention_analysis': attention_results,
        'removal_effectiveness': removal_results
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n🎯 Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()