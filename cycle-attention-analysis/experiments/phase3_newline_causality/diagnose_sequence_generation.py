#!/usr/bin/env python3
"""
Diagnose Sequence Generation Issues
==================================

This script will help us understand why so few sequences are being generated
of each type in the attention fallback analysis.
"""

print("🔧 Starting diagnostic imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')
sys.path.append('/home/mmahaut/projects/parrots/cycle-attention-analysis/src')

import torch
from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer
from modules.cached_data_utils import load_text_dataset
from modules.model_generated_cycle_processor import ModelGeneratedCycleProcessor
print("✅ All imports successful!")

def diagnose_sequence_generation():
    print("🚀 Diagnosing sequence generation...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🔧 Using device: {device}")
    
    # Load model
    print("🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    model.to(device)
    model.eval()
    print("   ✅ Model ready!")
    
    # Load small dataset for testing
    print("📚 Loading test dataset...")
    texts = load_text_dataset(n_samples=100)  # Smaller sample for diagnosis
    print(f"   ✅ Loaded {len(texts)} texts")
    
    # Test sequence generation with different parameters
    print("🔄 Testing sequence generation...")
    cycle_processor = ModelGeneratedCycleProcessor(tokenizer)
    
    print("\n📊 Test 1: Current parameters (max_length=32, max_new_tokens=1000)")
    natural_seqs, icl_seqs, no_cycle_seqs, no_cycle_icl_seqs = cycle_processor.process_texts(
        texts[:20],  # Even smaller test
        model,
        n_cycles=3,
        max_length=32,
        max_new_tokens=1000,
        batch_size=4
    )
    
    print(f"   Results with current parameters:")
    print(f"     - Natural: {len(natural_seqs)}")
    print(f"     - ICL: {len(icl_seqs)}")
    print(f"     - No-Cycle: {len(no_cycle_seqs)}")
    print(f"     - No-Cycle-ICL: {len(no_cycle_icl_seqs)}")
    
    if len(natural_seqs) > 0:
        print(f"   ✅ Sample Natural sequence length: {len(natural_seqs[0]['sequence'])} tokens")
    if len(no_cycle_icl_seqs) > 0:
        print(f"   ✅ Sample No-Cycle-ICL sequence length: {len(no_cycle_icl_seqs[0]['sequence'])} tokens")
    
    print("\n📊 Test 2: More relaxed parameters (max_length=64, max_new_tokens=200)")
    natural_seqs2, icl_seqs2, no_cycle_seqs2, no_cycle_icl_seqs2 = cycle_processor.process_texts(
        texts[:20],
        model,
        n_cycles=3,
        max_length=64,
        max_new_tokens=200,
        batch_size=4
    )
    
    print(f"   Results with relaxed parameters:")
    print(f"     - Natural: {len(natural_seqs2)}")
    print(f"     - ICL: {len(icl_seqs2)}")
    print(f"     - No-Cycle: {len(no_cycle_seqs2)}")
    print(f"     - No-Cycle-ICL: {len(no_cycle_icl_seqs2)}")
    
    # Show some example sequences if available
    if len(natural_seqs2) > 0:
        print(f"\n📝 Example Natural sequence:")
        sample_tokens = tokenizer.convert_ids_to_tokens(natural_seqs2[0]['sequence'][:50])
        print(f"   {' '.join(sample_tokens)}")
    
    if len(no_cycle_icl_seqs2) > 0:
        print(f"\n📝 Example No-Cycle-ICL sequence:")
        sample_tokens = tokenizer.convert_ids_to_tokens(no_cycle_icl_seqs2[0]['sequence'][:50])
        print(f"   {' '.join(sample_tokens)}")
    else:
        print(f"\n❌ No No-Cycle-ICL sequences generated!")
        print("   This explains why we have 0 sequences in the analysis")
    
    print(f"\n🎯 Recommendations:")
    if len(natural_seqs2) < 10:
        print("   📉 Low Natural sequence generation - consider:")
        print("     • Increasing max_new_tokens")
        print("     • Relaxing cycle detection criteria")
        print("     • Using more input texts")
    
    if len(no_cycle_icl_seqs2) == 0:
        print("   🚨 CRITICAL: No No-Cycle-ICL sequences generated!")
        print("     • Check if the sequence type is correctly implemented")
        print("     • Verify the generation logic in ModelGeneratedCycleProcessor")
        print("     • May need to adjust generation criteria")

if __name__ == "__main__":
    diagnose_sequence_generation()