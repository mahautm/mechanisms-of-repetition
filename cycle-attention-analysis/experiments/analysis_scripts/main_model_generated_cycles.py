import typer
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add the parrots project to path
sys.path.append('/home/mmahaut/projects/parrots')

from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from modules.attention_analyzer_fixed import AttentionAnalyzerFixed
from modules.model_generated_cycle_processor import ModelGeneratedCycleProcessor
from modules.visualization import AttentionVisualizer

def safe_mkdir(path):
    """Safely create directory, handling conflicts."""
    path = Path(path)
    
    if path.exists() and not path.is_dir():
        print(f"Warning: {path} exists as a file, removing it")
        path.unlink()
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        raise

def main(
    model_name: str = "EleutherAI/pythia-1.4b",
    revision: str = None,
    layer: int = 12,
    n_cycles: int = 3,
    use_bfloat16: bool = False,
    seed: int = 42,
    batch_size: int = 4,
    max_length: int = 256,
    max_new_tokens: int = 100,
    n_samples: int = 200,
    output_dir: str = "../data/results"
):
    """
    Analyze attention patterns for MODEL-GENERATED cycles with 4 sequence types.
    """
    print(f"Starting MODEL-GENERATED cycle attention analysis for layer {layer}")
    print(f"Using {n_samples} samples with {n_cycles} cycles")
    print("Four sequence types:")
    print("1. Natural: Model-generated cycles extended naturally")
    print("2. ICL: Detected cycles repeated artificially")  
    print("3. No-cycle: Sequences without any cycles")
    print("4. No-cycle ICL: Non-repetitive patterns repeated artificially")
    
    # Setup output directory
    safe_mkdir(output_dir)
    output_path = Path(output_dir)
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16)
        model.eval()
        device = get_device()
        model.to(device)
        
        print(f"Loaded model {model_name}")
        
        # Load data
        print("Loading test dataset...")
        texts = load_text_dataset("JeanKaddour/minipile", seed=seed, n_samples=n_samples)
        print(f"Loaded {len(texts)} text samples")
        
        # Initialize components
        cycle_processor = ModelGeneratedCycleProcessor(tokenizer)
        attention_analyzer = AttentionAnalyzerFixed(device, tokenizer)
        visualizer = AttentionVisualizer(output_path)
        
        # Process data: generate with model first, then detect cycles in generation
        print("Generating with model and detecting cycles in output...")
        natural_sequences, icl_sequences, no_cycle_sequences, no_cycle_icl_sequences = cycle_processor.process_texts(
            texts, model, n_cycles, max_length, max_new_tokens, batch_size
        )
        
        print(f"\nFound sequences:")
        print(f"- {len(natural_sequences)} with model-generated cycles (natural)")
        print(f"- {len(icl_sequences)} ICL sequences (detected cycles repeated)")
        print(f"- {len(no_cycle_sequences)} without cycles")
        print(f"- {len(no_cycle_icl_sequences)} no-cycle ICL (non-repetitive patterns repeated)")
        
        # Show examples of what we found
        if natural_sequences:
            print("\nExample of natural model-generated cycle:")
            example = natural_sequences[0]
            print(f"  Original: '{example['original_text'][:60]}...'")
            print(f"  Generated cycle: '{example['cycle_text']}'")
            print(f"  Repeated {example['n_cycles']} times naturally")
        
        if no_cycle_icl_sequences:
            print("\nExample of no-cycle ICL (forced repetition):")
            example = no_cycle_icl_sequences[0]
            print(f"  Original: '{example['original_text'][:60]}...'")
            print(f"  Pattern that was NOT repetitive: '{example['cycle_text']}'")
            print(f"  Artificially repeated {example['n_cycles']} times")
        
        # Analyze attention patterns for each type
        results = {}
        
        if natural_sequences:
            print("\nAnalyzing natural sequences with model-generated cycles...")
            results['natural'] = attention_analyzer.analyze_attention_patterns(
                natural_sequences, model, tokenizer, "natural"
            )
            print(f"✅ Natural analysis: {len(results['natural']['head_statistics'])} layers")
        
        if icl_sequences:
            print("Analyzing ICL sequences...")  
            results['icl'] = attention_analyzer.analyze_attention_patterns(
                icl_sequences, model, tokenizer, "icl"
            )
            print(f"✅ ICL analysis: {len(results['icl']['head_statistics'])} layers")
        
        if no_cycle_sequences:
            print("Analyzing no-cycle sequences...")
            results['no_cycle'] = attention_analyzer.analyze_attention_patterns(
                no_cycle_sequences, model, tokenizer, "JeanKaddour/minipile"
            )
            print(f"✅ No-cycle analysis: {len(results['no_cycle']['head_statistics'])} layers")
        
        if no_cycle_icl_sequences:
            print("Analyzing no-cycle ICL sequences...")
            results['no_cycle_icl'] = attention_analyzer.analyze_attention_patterns(
                no_cycle_icl_sequences, model, tokenizer, "no_cycle_icl"
            )
            print(f"✅ No-cycle ICL analysis: {len(results['no_cycle_icl']['head_statistics'])} layers")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        try:
            visualizer.create_all_plots(results, layer)
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
        
        # Save results
        results_file = output_path / f"model_generated_cycles_layer_{layer}.pt"
        torch.save(results, results_file)
        print(f"Results saved to {results_file}")
        
        print(f"\n🎉 MODEL-GENERATED cycle analysis complete for layer {layer}!")
        print(f"This gives you 4 different conditions to compare attention patterns:")
        print(f"1. Natural cycles (model chose to repeat)")
        print(f"2. ICL cycles (we repeated what model chose to repeat)")
        print(f"3. No cycles (model chose not to repeat)")
        print(f"4. No-cycle ICL (we repeated what model chose NOT to repeat)")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    typer.run(main)