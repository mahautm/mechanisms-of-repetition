print("🔧 Starting imports...")
import sys
sys.path.append('/home/mmahaut/projects/parrots')

import torch
print("✅ torch imported")
from tqdm import tqdm
import time
import argparse
print("✅ Basic imports done")

try:
    from parrots.aa_fortu.modules.model_utils import load_model_and_tokenizer, get_device
    print("✅ model_utils imported")
except ImportError as e:
    print(f"❌ Failed to import model_utils: {e}")
    raise

try:
    from modules.cached_data_utils import load_text_dataset
    print("✅ cached_data_utils imported")
except ImportError as e:
    print(f"❌ Failed to import cached_data_utils: {e}")
    raise

try:
    from modules.model_generated_cycle_processor import ModelGeneratedCycleProcessor
    print("✅ model_generated_cycle_processor imported")
except ImportError as e:
    print(f"❌ Failed to import model_generated_cycle_processor: {e}")
    raise

try:
    from modules.cycle_evolution_plotter import CycleEvolutionPlotter
    print("✅ cycle_evolution_plotter imported")
except ImportError as e:
    print(f"❌ Failed to import cycle_evolution_plotter: {e}")
    raise

from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
print("✅ All imports successful!")

def plot_focus_token_evolution(focus_tokens_data, seq_type, target_layer, max_cycles, save_path):
    """Plot evolution of most focused tokens across cycles for each head."""
    if not focus_tokens_data:
        return
    
    n_heads = len(focus_tokens_data[0]) if focus_tokens_data else 0
    if n_heads == 0:
        return
    
    # Create subplots for each head
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))  # Assuming 16 heads max
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for head_idx in range(min(n_heads, 16)):  # Limit to 16 heads for visualization
        ax = axes[head_idx]
        
        # Collect token frequencies across all sequences for this head
        token_counts_by_cycle = defaultdict(lambda: defaultdict(int))
        
        for seq_focus_tokens in focus_tokens_data:
            if head_idx < len(seq_focus_tokens):
                head_tokens = seq_focus_tokens[head_idx]
                for token_info in head_tokens:
                    cycle = token_info['cycle']
                    token = token_info['token']
                    token_counts_by_cycle[cycle][token] += 1
        
        # Plot most common tokens per cycle
        cycles = sorted(token_counts_by_cycle.keys())
        for cycle in cycles:
            token_counts = token_counts_by_cycle[cycle]
            most_common = Counter(token_counts).most_common(3)  # Top 3 tokens
            
            for i, (token, count) in enumerate(most_common):
                ax.scatter(cycle, i, s=count*10, alpha=0.7, label=f"{token}" if cycle == cycles[0] else "")
        
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Token Rank')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Focus Token Evolution - {seq_type} (Layer {target_layer})')
    plt.tight_layout()
    
    try:
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Focus plot saved successfully to: {save_path}")
    except Exception as e:
        print(f"   ❌ Error saving focus plot to {save_path}: {e}")
        # Try fallback location
        fallback_path = Path("/tmp") / save_path.name
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(f"   🚨 Focus plot saved to fallback: {fallback_path}")
    finally:
        plt.close()

def plot_attention_distribution_evolution(attention_dist_data, seq_type, target_layer, max_cycles, save_path):
    """Plot evolution of attention distribution across segments for each head."""
    if not attention_dist_data:
        return
    
    n_heads = len(attention_dist_data[0]) if attention_dist_data else 0
    if n_heads == 0:
        return
    
    # Create subplots for each head
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for head_idx in range(min(n_heads, 16)):
        ax = axes[head_idx]
        
        # Aggregate attention distributions across sequences
        cycle_distributions = defaultdict(lambda: defaultdict(list))
        
        for seq_distributions in attention_dist_data:
            if head_idx < len(seq_distributions):
                head_distributions = seq_distributions[head_idx]
                for dist_info in head_distributions:
                    cycle_end = dist_info['cycle_end']
                    segments = dist_info['segments']
                    
                    # Store prompt attention
                    cycle_distributions[cycle_end]['prompt'].append(segments['prompt'])
                    
                    # Store cycle attentions
                    for cycle_name, attention in segments['cycles'].items():
                        cycle_distributions[cycle_end][cycle_name].append(attention)
        
        # Plot mean attention distributions
        cycles = sorted(cycle_distributions.keys())
        segment_names = ['prompt'] + [f'cycle_{i+1}' for i in range(max_cycles)]
        
        for segment_name in segment_names:
            means = []
            stds = []
            for cycle in cycles:
                values = cycle_distributions[cycle][segment_name]
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.errorbar(cycles, means, yerr=stds, label=segment_name, marker='o', alpha=0.7)
        
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Cycle End')
        ax.set_ylabel('Attention Weight')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Attention Distribution Evolution - {seq_type} (Layer {target_layer})')
    plt.tight_layout()
    
    try:
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Attention plot saved successfully to: {save_path}")
    except Exception as e:
        print(f"   ❌ Error saving attention plot to {save_path}: {e}")
        # Try fallback location
        fallback_path = Path("/tmp") / save_path.name
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(f"   🚨 Attention plot saved to fallback: {fallback_path}")
    finally:
        plt.close()

def analyze_cycle_evolution_parametric(max_cycles, target_layer, n_samples=1000, sequence_type="all", all_heads=True, checkpoint=None):
    """
    Parametric version: analyze specific cycle length and layer.
    
    Args:
        max_cycles: Number of cycles to analyze (3, 5, 7, etc.)
        target_layer: Specific layer to analyze (0-23)
        sequence_type: "natural", "icl", "no_cycle_icl", or "all"
        all_heads: Whether to analyze all heads or just representative ones
    """
    
    print(f"🚀 FUNCTION CALLED: analyze_cycle_evolution_parametric")
    print(f"=== Parametric Cycle Evolution Analysis ===")
    print(f"Parameters: cycles={max_cycles}, layer={target_layer}, seq_type={sequence_type}, all_heads={all_heads}")
    
    # Debug: Show current working directory
    current_dir = os.getcwd()
    print(f"📁 Current working directory: {current_dir}")
    
    # Start timing
    start_time = time.time()
    
    # Load model
    print("🤖 Loading model...")
    # Try CUDA first, fall back to CPU if unavailable
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"   🔧 Using device: {device}")
            print(f"   � CUDA device count: {torch.cuda.device_count()}")
        else:
            device = torch.device('cpu')
            print(f"   🔧 Using device: {device} (CUDA not available)")
    except Exception as e:
        device = torch.device('cpu')
        print(f"   ⚠️  CUDA error, falling back to CPU: {e}")
    
    print("📦 About to load model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer("EleutherAI/pythia-1.4b", revision=checkpoint)
    print("✅ Model and tokenizer loaded!")
    
    try:
        model.to(device)
        print(f"   ✅ Model moved to: {device}")
    except Exception as e:
        print(f"   ⚠️  Failed to move model to {device}, using CPU: {e}")
        device = torch.device('cpu')
        model.to(device)
        
    model.eval()
    print("✅ Model set to eval mode!")
    
    # Verify model is on GPU
    if next(model.parameters()).device.type == 'cuda':
        print(f"   ✅ Model loaded on GPU: {next(model.parameters()).device}")
    else:
        print(f"   ⚠️  Model loaded on CPU: {next(model.parameters()).device}")
        print(f"   ⚠️  Warning: Running on CPU will be very slow!")
        
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'compile'):
        print("   🚀 Enabling torch.compile for acceleration...")
        # model = torch.compile(model)  # Uncomment if using PyTorch 2.0+
    
        # Load dataset with full size for comprehensive analysis
    print("📚 Loading dataset...")
    print(f"📊 Requested n_samples: {n_samples}")
    texts = load_text_dataset(n_samples=n_samples)  # Full dataset for comprehensive results
    print(f"✅ Loaded {len(texts)} texts")
    
    print("🔄 Processing texts to find model-generated cycles...")
    cycle_processor = ModelGeneratedCycleProcessor(tokenizer)
    
    # Process all sequence types once
    step_start = time.time()
    natural_seqs, icl_seqs, no_cycle_seqs, no_cycle_icl_seqs = cycle_processor.process_texts(
        texts, model, 
        n_cycles=max_cycles,
        max_length=32,
        max_new_tokens=1000,
        batch_size=32
    )
    
    processing_time = time.time() - step_start
    print(f"   ✅ Cycle processing complete in {processing_time:.1f}s")
    
    # Select which sequences to analyze
    sequences_to_analyze = {}
    if sequence_type == "all":
        sequences_to_analyze = {
            "natural": natural_seqs,
            "icl": icl_seqs,
            "no_cycle_icl": no_cycle_icl_seqs,
            "JeanKaddour/minipile": no_cycle_seqs
        }
    elif sequence_type == "natural":
        sequences_to_analyze = {"natural": natural_seqs}
    elif sequence_type == "icl":
        sequences_to_analyze = {"icl": icl_seqs}
    elif sequence_type == "no_cycle_icl":
        sequences_to_analyze = {"no_cycle_icl": no_cycle_icl_seqs}
    elif sequence_type == "JeanKaddour/minipile":
        sequences_to_analyze = {"JeanKaddour/minipile": no_cycle_seqs}
    
    print(f"📊 Analyzing sequences:")
    for seq_type, seqs in sequences_to_analyze.items():
        print(f"  - {seq_type}: {len(seqs)} sequences")
    
    # 1) Get token which is most focused on by each attention head in the target layer (at the end of each cycle)
    # 2) Track how this focus token changes over cycles
    # 3) Plot the evolution of focus tokens over cycles for each head

    # 4) take the attention matrix at the end of each cycle for the each head in the target layer
    # 5) look at how much of the distribution is focused on the prompt, v.s. the first cycle, second cycle, etc.
    # 6) plot how the attention distribution changes over cycles for each head

    # Initialize results storage
    all_results = {}
    
    # Process each sequence type
    for seq_type, sequences in sequences_to_analyze.items():
        print(f"\n🔍 Analyzing {seq_type} sequences...")
        
        if not sequences:
            print(f"   ⚠️  No sequences found for {seq_type}, skipping...")
            continue
            
        seq_results = {
            'focus_tokens': [],  # Most focused tokens per head per cycle
            'attention_distributions': [],  # Attention distribution across segments
            'cycle_positions': [],  # Positions of cycle ends
            'sequences': []  # Store sequences for reference
        }
        
        # Analyze each sequence
        for seq_idx, seq_data in enumerate(tqdm(sequences[:50], desc=f"Processing {seq_type}")):  # Limit for performance
            try:
                # Get model outputs with attention
                with torch.no_grad():
                    # seq_data['sequence'] is already tokenized (list of token IDs)
                    input_ids = torch.tensor([seq_data['sequence']]).to(device)
                    inputs = {'input_ids': input_ids}
                    
                    # Truncate if too long
                    if input_ids.shape[1] > 1024:
                        inputs['input_ids'] = input_ids[:, :1024]
                    
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions[target_layer]  # Shape: (1, n_heads, seq_len, seq_len)
                    
                # Get cycle end positions
                cycle_ends = []
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # Find cycle boundaries (simplified - you may need to adjust based on your cycle structure)
                for i in range(1, max_cycles + 1):
                    # This is a simplified approach - adjust based on your cycle detection logic
                    cycle_end_pos = min(len(tokens) - 1, i * (len(tokens) // (max_cycles + 1)))
                    cycle_ends.append(cycle_end_pos)
                
                # 1) & 2) Get most focused token for each head at each cycle end
                focus_tokens_per_head = []
                for head_idx in range(attentions.shape[1]):
                    head_focus_tokens = []
                    for cycle_end_pos in cycle_ends:
                        if cycle_end_pos < attentions.shape[2]:
                            # Get attention weights from the cycle end position
                            attn_weights = attentions[0, head_idx, cycle_end_pos, :]
                            most_focused_pos = torch.argmax(attn_weights).item()
                            most_focused_token = tokens[most_focused_pos] if most_focused_pos < len(tokens) else "<UNK>"
                            head_focus_tokens.append({
                                'cycle': len(head_focus_tokens) + 1,
                                'position': most_focused_pos,
                                'token': most_focused_token,
                                'attention_weight': attn_weights[most_focused_pos].item()
                            })
                    focus_tokens_per_head.append(head_focus_tokens)
                
                # 4) & 5) Get attention distribution across segments for each head
                attention_distributions_per_head = []
                for head_idx in range(attentions.shape[1]):
                    head_distributions = []
                    for cycle_idx, cycle_end_pos in enumerate(cycle_ends):
                        if cycle_end_pos < attentions.shape[2]:
                            attn_weights = attentions[0, head_idx, cycle_end_pos, :]
                            
                            # Define segments (adjust based on your sequence structure)
                            prompt_end = seq_data.get('prompt_length', len(tokens) // 4)
                            cycle_length = (len(tokens) - prompt_end) // max_cycles if max_cycles > 0 else 0
                            
                            # Calculate attention distribution across segments
                            segments = {
                                'prompt': attn_weights[:prompt_end].sum().item() if prompt_end > 0 else 0.0,
                                'cycles': {}
                            }
                            
                            for i in range(max_cycles):
                                start_pos = prompt_end + i * cycle_length
                                end_pos = min(prompt_end + (i + 1) * cycle_length, len(tokens))
                                if start_pos < len(tokens) and end_pos > start_pos:
                                    segments['cycles'][f'cycle_{i+1}'] = attn_weights[start_pos:end_pos].sum().item()
                                else:
                                    segments['cycles'][f'cycle_{i+1}'] = 0.0
                            
                            head_distributions.append({
                                'cycle_end': cycle_idx + 1,
                                'segments': segments
                            })
                    attention_distributions_per_head.append(head_distributions)
                
                # Store results for this sequence
                seq_results['focus_tokens'].append(focus_tokens_per_head)
                seq_results['attention_distributions'].append(attention_distributions_per_head)
                seq_results['cycle_positions'].append(cycle_ends)
                seq_results['sequences'].append(seq_data)
                
            except Exception as e:
                print(f"   ⚠️  Error processing sequence {seq_idx}: {e}")
                continue
        
        all_results[seq_type] = seq_results
        print(f"   ✅ Processed {len(seq_results['sequences'])} {seq_type} sequences")
    
    # 3) & 6) Create plots for the evolution analysis
    print(f"\n📈 Creating evolution plots...")
    
    # Use absolute path to ensure we know where files go
    base_dir = Path(current_dir)
    output_dir = base_dir / "plots" / "cycle_evolution_parametric" / f"cycles_{max_cycles}" / ('steplatest' if checkpoint is None else checkpoint)
    
    print(f"📂 Output directory (absolute): {output_dir.absolute()}")
    
    # Ensure directory exists with robust error handling
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory created/verified: {output_dir.absolute()}")
        
        # Verify we can write to the directory
        test_file = output_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()  # Delete test file
        print(f"✅ Write permissions confirmed")
        
    except Exception as e:
        print(f"❌ ERROR creating output directory: {e}")
        # Fallback to a safe location
        fallback_dir = Path("/tmp") / f"cycle_evolution_backup_{max_cycles}_{target_layer}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        output_dir = fallback_dir
        print(f"🚨 Using fallback directory: {output_dir.absolute()}")
    for seq_type, results in all_results.items():
        if not results['sequences']:
            continue
        
        # Plot focus token evolution
        focus_plot_path = output_dir / f"{seq_type}_c{max_cycles}_l{target_layer}_focus_evolution_layer_{target_layer}.png"
        print(f"📊 About to save focus plot to: {focus_plot_path.absolute()}")
        
        try:
            plot_focus_token_evolution(
                results['focus_tokens'], seq_type, target_layer, max_cycles, focus_plot_path
            )
            # Verify file was actually created
            if focus_plot_path.exists():
                file_size = focus_plot_path.stat().st_size
                print(f"   ✅ Saved focus token evolution plot: {focus_plot_path.absolute()} ({file_size} bytes)")
            else:
                print(f"   ❌ Plot file was not created: {focus_plot_path.absolute()}")
        except Exception as e:
            print(f"   ❌ Error saving focus plot: {e}")
        
        # Plot attention distribution evolution
        attention_plot_path = output_dir / f"{seq_type}_c{max_cycles}_l{target_layer}_attention_evolution_layer_{target_layer}.png"
        print(f"📊 About to save attention plot to: {attention_plot_path.absolute()}")
        
        try:
            plot_attention_distribution_evolution(
                results['attention_distributions'], seq_type, target_layer, max_cycles, attention_plot_path
            )
            # Verify file was actually created
            if attention_plot_path.exists():
                file_size = attention_plot_path.stat().st_size
                print(f"   ✅ Saved attention distribution evolution plot: {attention_plot_path.absolute()} ({file_size} bytes)")
            else:
                print(f"   ❌ Plot file was not created: {attention_plot_path.absolute()}")
        except Exception as e:
            print(f"   ❌ Error saving attention plot: {e}")

    
    # Print summary statistics
    print(f"\n📊 Summary Results:")
    for seq_type, results in all_results.items():
        print(f"  {seq_type}:")
        print(f"    - Sequences analyzed: {len(results['sequences'])}")
        print(f"    - Average cycles per sequence: {max_cycles}")
        print(f"    - Target layer: {target_layer}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total analysis time: {total_time:.1f}s")
    # save results
    results_path = output_dir / f"cycle_evolution_parametric_c{max_cycles}_l{target_layer}_{sequence_type}_results.pt"
    print(f"💾 About to save results to: {results_path.absolute()}")
    
    try:
        torch.save(all_results, results_path)
        # Verify file was actually saved
        if results_path.exists():
            file_size = results_path.stat().st_size
            print(f"   ✅ Data saved: {results_path.absolute()} ({file_size} bytes)")
        else:
            print(f"   ❌ Results file was not created: {results_path.absolute()}")
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")
        # Try saving to backup location
        backup_path = Path("/tmp") / f"backup_results_c{max_cycles}_l{target_layer}.pt"
        torch.save(all_results, backup_path)
        print(f"   � Backup saved to: {backup_path.absolute()}")
    
    print(f"�📁 All files saved to directory: {output_dir.absolute()}")
    
    # List all files in the output directory for verification
    if output_dir.exists():
        files = list(output_dir.glob('*'))
        print(f"📋 Files in output directory ({len(files)} total):")
        for file in files:
            print(f"   - {file.name} ({file.stat().st_size} bytes)")
    else:
        print(f"❌ Output directory does not exist: {output_dir.absolute()}")
    
    return all_results

if __name__ == "__main__":
    print("🚀 Script starting - parsing arguments...")
    
    parser = argparse.ArgumentParser(description="Parametric Cycle Evolution Analysis")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cycles to analyze")
    parser.add_argument("--layer", type=int, default=10, help="Target layer to analyze")
    parser.add_argument("--seq_type", type=str, default="all", 
                       choices=["natural", "icl", "no_cycle_icl", "JeanKaddour/minipile", "all"],
                       help="Sequence type to analyze")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--all_heads", action="store_true", default=True, help="Analyze all heads")
    parser.add_argument("--revision", type=str, default=None, help="Model checkpoint/revision")
    
    try:
        args = parser.parse_args()
        print(f"✅ Arguments parsed successfully:")
        print(f"   - cycles: {args.cycles}")
        print(f"   - layer: {args.layer}")
        print(f"   - seq_type: {args.seq_type}")
        print(f"   - n_samples: {args.n_samples}")
        print(f"   - all_heads: {args.all_heads}")
        print(f"   - revision: {args.revision}")
        
        print("🔄 About to call analyze_cycle_evolution_parametric...")
        
        # Call the main function with parsed arguments
        result = analyze_cycle_evolution_parametric(
            max_cycles=args.cycles,
            target_layer=args.layer,
            n_samples=args.n_samples,
            sequence_type=args.seq_type,
            all_heads=args.all_heads,
            checkpoint=args.revision
        )
        
        print("✅ Function completed successfully!")
        print(f"📊 Results keys: {list(result.keys()) if result else 'None'}")
        
    except Exception as e:
        print(f"❌ ERROR in main: {e}")
        import traceback
        traceback.print_exc()
        raise
