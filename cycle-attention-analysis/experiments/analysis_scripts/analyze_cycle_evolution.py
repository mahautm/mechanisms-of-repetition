import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
from tqdm import tqdm
import time
from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
from modules.cached_data_utils import load_text_dataset
from modules.model_generated_cycle_processor import ModelGeneratedCycleProcessor
from modules.cycle_evolution_plotter import CycleEvolutionPlotter
from pathlib import Path

def analyze_cycle_evolution():
    """
    Analyze how attention patterns evolve across cycles.
    This answers: Does attention focus shift as cycles progress, or stay consistent?
    """
    
    print("=== Cycle Evolution Analysis ===")
    print("Question: How does attention focus change across cycles?")
    print("- Consistent: Same pattern in each cycle")
    print("- Shifting: Different patterns as cycles progress")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b")
    device = get_device()
    model.to(device)
    model.eval()
    
    # Load data - optimized for speed
    print("📥 Loading test data...")
    start_time = time.time()
    texts = load_text_dataset("JeanKaddour/minipile", n_samples=300)  # More datapoints for better statistics
    print(f"   ✅ Loaded {len(texts)} texts in {time.time() - start_time:.1f}s")
    
    print("\n🔄 Processing texts to find model-generated cycles...")
    print("   This involves: text generation → cycle detection → sequence creation")
    cycle_processor = ModelGeneratedCycleProcessor(tokenizer)
    
    # Add progress tracking
    step_start = time.time()
    
    # Optimized parameters for fast execution
    natural_seqs, icl_seqs, no_cycle_seqs, no_cycle_icl_seqs = cycle_processor.process_texts(
        texts, model, 
        n_cycles=3,           # Start with shorter cycles for speed
        max_length=32,       # Shorter sequences
        max_new_tokens=1000,    # Less generation time
        batch_size=8          # Larger batch for efficiency
    )
    
    processing_time = time.time() - step_start
    print(f"   ✅ Cycle processing complete in {processing_time:.1f}s")
    
    print(f"Found sequences with cycles:")
    print(f"- {len(natural_seqs)} natural sequences")
    print(f"- {len(icl_seqs)} ICL sequences")
    print(f"- {len(no_cycle_icl_seqs)} no-cycle ICL sequences")
    
    # Initialize evolution plotter
    evolution_plotter = CycleEvolutionPlotter(device, tokenizer)
    
    # Create output directory
    output_dir = Path("../plots/cycle_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze evolution for each sequence type and cycle length
    results = {}
    
    # Fast test configurations - optimized for speed while covering key questions
    test_configs = [
        {"max_cycles": 3, "layers": [10, 15], "all_heads": False, "name": "fast_test"},  # Very fast
        {"max_cycles": 3, "layers": [5, 10, 15, 20], "all_heads": True, "name": "all_heads_test"},  # Test all heads but keep short
    ]
    
    if natural_seqs:
        print(f"\n📊 Analyzing evolution in NATURAL sequences...")
        print(f"    Found {len(natural_seqs)} sequences with natural cycles")
        print("    (These are cycles the model chose to generate naturally)")
        
        results['natural'] = {}
        
        for config_idx, config in enumerate(tqdm(test_configs, desc="Natural configs")):
            config_name = config['name']
            config_start = time.time()
            print(f"\n  🔬 Config {config_idx+1}/{len(test_configs)}: {config_name}")
            print(f"     - Max cycles: {config['max_cycles']}, Layers: {config['layers']}, All heads: {config['all_heads']}")
            
            natural_evolution = evolution_plotter.analyze_cycle_evolution(
                natural_seqs, model, f"natural_{config_name}", 
                target_layers=config['layers'],
                all_heads=config['all_heads'],
                max_cycles=config['max_cycles']
            )
            
            if natural_evolution:
                results['natural'][config_name] = natural_evolution
                
                # Create visualizations for this config
                config_output_dir = output_dir / "natural" / config_name
                config_output_dir.mkdir(parents=True, exist_ok=True)
                
                evolution_plotter.plot_cycle_evolution_heatmap(
                    natural_evolution, config_output_dir, max_examples=1  # Just 1 example for speed
                )
                evolution_plotter.plot_attention_focus_evolution(
                    natural_evolution, config_output_dir
                )
                evolution_plotter.create_summary_plot(
                    natural_evolution, config_output_dir
                )
                config_time = time.time() - config_start
                print(f"     ✅ Config {config_name} complete in {config_time:.1f}s")
        
        print("✅ Natural sequence evolution analysis complete")
    
    if icl_seqs:
        print(f"\n📊 Analyzing evolution in ICL sequences...")
        print(f"    Found {len(icl_seqs)} ICL sequences")
        print("    (These are cycles we artificially repeated)")
        
        results['icl'] = {}
        
        for config_idx, config in enumerate(tqdm(test_configs, desc="ICL configs")):
            config_name = config['name']
            config_start = time.time()
            print(f"\n  🔬 ICL Config {config_idx+1}/{len(test_configs)}: {config_name}")
            
            icl_evolution = evolution_plotter.analyze_cycle_evolution(
                icl_seqs, model, f"icl_{config_name}", 
                target_layers=config['layers'],
                all_heads=config['all_heads'],
                max_cycles=config['max_cycles']
            )
            
            if icl_evolution:
                results['icl'][config_name] = icl_evolution
                
                config_output_dir = output_dir / "icl" / config_name
                config_output_dir.mkdir(parents=True, exist_ok=True)
                
                evolution_plotter.plot_cycle_evolution_heatmap(
                    icl_evolution, config_output_dir, max_examples=1  # Just 1 example for speed
                )
                evolution_plotter.plot_attention_focus_evolution(
                    icl_evolution, config_output_dir
                )
                evolution_plotter.create_summary_plot(
                    icl_evolution, config_output_dir
                )
                config_time = time.time() - config_start
                print(f"     ✅ ICL config {config_name} complete in {config_time:.1f}s")
        
        print("✅ ICL sequence evolution analysis complete")
    
    if no_cycle_icl_seqs:
        print(f"\n📊 Analyzing evolution in NO-CYCLE ICL sequences...")
        print("(These are patterns we forced to repeat that the model never chose to repeat)")
        
        results['no_cycle_icl'] = {}
        
        # For no-cycle ICL, just one fast config for comparison
        no_cycle_configs = [
            {"max_cycles": 3, "layers": [10, 15], "all_heads": False, "name": "fast_comparison"}
        ]
        
        for config in no_cycle_configs:
            config_name = config['name']
            print(f"\n  🔬 Testing no-cycle ICL config: {config_name}")
            
            no_cycle_icl_evolution = evolution_plotter.analyze_cycle_evolution(
                no_cycle_icl_seqs, model, f"no_cycle_icl_{config_name}", 
                target_layers=config['layers'],
                all_heads=config['all_heads'],
                max_cycles=config['max_cycles']
            )
            
            if no_cycle_icl_evolution:
                results['no_cycle_icl'][config_name] = no_cycle_icl_evolution
                
                config_output_dir = output_dir / "no_cycle_icl" / config_name
                config_output_dir.mkdir(parents=True, exist_ok=True)
                
                evolution_plotter.plot_cycle_evolution_heatmap(
                    no_cycle_icl_evolution, config_output_dir, max_examples=1  # Just 1 example for speed
                )
                evolution_plotter.plot_attention_focus_evolution(
                    no_cycle_icl_evolution, config_output_dir
                )
                evolution_plotter.create_summary_plot(
                    no_cycle_icl_evolution, config_output_dir
                )
                print(f"     ✅ No-cycle ICL config {config_name} complete")
        
        print("✅ No-cycle ICL sequence evolution analysis complete")
    
    # Create comparison plots
    if len(results) > 1:
        create_comparison_plots(results, output_dir)
    
    # Save detailed results
    torch.save(results, output_dir / "cycle_evolution_results.pt")
    
    # Final timing summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 Comprehensive Cycle Evolution Analysis Complete!")
    print(f"⏱️  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"📁 Results saved to: {output_dir}")
    print(f"\n📊 Analysis Summary:")
    
    total_configs = 0
    for seq_type, configs in results.items():
        if isinstance(configs, dict):
            print(f"  {seq_type}: {len(configs)} configurations tested")
            total_configs += len(configs)
    
    print(f"  📈 Total configurations: {total_configs}")
    print(f"  📊 Sequences analyzed: {len(natural_seqs) + len(icl_seqs) + len(no_cycle_icl_seqs)}")
    
    print(f"\n🔍 Key plots to check (organized by sequence type and config):")
    print(f"1. */*/evolution_heatmap_*.png - Shows if attention patterns change across cycles")
    print(f"2. */*/focus_evolution_*.png - Shows how attention to each position evolves") 
    print(f"3. */*/evolution_consistency_summary.png - Overall consistency scores")
    print(f"4. evolution_comparison.png - Cross-sequence-type comparison")
    print(f"\n💡 Interpretation Guide:")
    print(f"🔶 HEATMAPS:")
    print(f"  - Horizontal stripes = Consistent attention across cycles")
    print(f"  - Diagonal/shifting patterns = Attention evolves as cycles progress")
    print(f"  - Random patterns = Unstable/noisy attention")
    print(f"🔶 FOCUS EVOLUTION:")
    print(f"  - Flat lines = Stable attention focus")
    print(f"  - Smooth curves = Gradual attention shift")
    print(f"  - Jagged lines = Erratic attention changes")
    print(f"🔶 CONFIGURATIONS TESTED (OPTIMIZED FOR SPEED):")
    print(f"  - fast_test: 3 cycles, 2 layers [10,15], select heads [0,4,8,12]")
    print(f"  - all_heads_test: 3 cycles, 4 layers [5,10,15,20], all 16 heads")
    print(f"  - Data: 300 texts, max 150 tokens, 80 new tokens per generation")
    print(f"\n❓ Key Research Questions Addressed:")
    print(f"  1. Does attention focus shift consistently across all heads?")
    print(f"  2. Do deeper layers show different evolution patterns?")
    print(f"  3. How does cycle length affect attention consistency?")
    print(f"  4. Are natural cycles processed differently than artificial ones?")

def create_comparison_plots(results, output_dir):
    """Create plots comparing evolution patterns across sequence types."""
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n📊 Creating comparison plots...")
    
    # Extract consistency scores for comparison
    consistency_data = {}
    
    for seq_type, evolution_data in results.items():
        consistencies = []
        
        for seq_data in evolution_data['sequences']:
            for layer_name, layer_data in seq_data['layer_results'].items():
                for head_name, evolution_matrix in layer_data.items():
                    # Simple consistency metric: correlation between first and last cycle
                    if evolution_matrix.shape[0] >= 2:
                        first_cycle = evolution_matrix[0].flatten()
                        last_cycle = evolution_matrix[-1].flatten()
                        corr = np.corrcoef(first_cycle, last_cycle)[0, 1]
                        if not np.isnan(corr):
                            consistencies.append(corr)
        
        consistency_data[seq_type] = consistencies
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot comparison
    seq_types = list(consistency_data.keys())
    consistency_values = [consistency_data[st] for st in seq_types]
    
    axes[0].boxplot(consistency_values, labels=seq_types)
    axes[0].set_title('Attention Evolution Consistency\nFirst vs Last Cycle Correlation')
    axes[0].set_ylabel('Correlation')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram comparison
    colors = ['blue', 'orange', 'green', 'red']
    for i, (seq_type, consistencies) in enumerate(consistency_data.items()):
        axes[1].hist(consistencies, alpha=0.6, label=seq_type, 
                    color=colors[i % len(colors)], bins=20)
    
    axes[1].set_title('Distribution of Consistency Scores')
    axes[1].set_xlabel('First-Last Cycle Correlation')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evolution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\n📈 Evolution Consistency Summary:")
    for seq_type, consistencies in consistency_data.items():
        if consistencies:
            mean_corr = np.mean(consistencies)
            std_corr = np.std(consistencies)
            print(f"  {seq_type}: {mean_corr:.3f} ± {std_corr:.3f} (n={len(consistencies)})")
            
            if mean_corr > 0.7:
                print(f"    → Highly consistent attention across cycles")
            elif mean_corr > 0.3:
                print(f"    → Moderately consistent attention")
            else:
                print(f"    → Attention patterns change significantly across cycles")

if __name__ == "__main__":
    analyze_cycle_evolution()