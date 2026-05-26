#!/usr/bin/env python3
"""
OLMo Attention Analysis Pipeline
Adapted from ckpt_pipeline_main.py for OLMo models
Key changes:
- Uses 'model.layers.{layer}' instead of 'gpt_neox.layers.{layer}'
- No revision/checkpoint (OLMo doesn't have intermediate checkpoints)
- Adapted for 16 layers instead of 24
"""
import typer
import numpy as np
from pathlib import Path
import sys
sys.path.append('/home/mmahaut/projects/parrots')

from parrots.aa_fortu.modules.model_utils import HookedModel, load_model_and_tokenizer, get_device
from parrots.aa_fortu.aa_fortu_train_lens import Lens
from parrots.aa_fortu.aa_fortu_train_multihead_lens import MultiHeadLens
from parrots.aa_fortu.modules.data_utils import load_text_dataset, pretokenize_texts
from parrots.aa_fortu.modules.contrast_analysis import extract_contrasts
from parrots.aa_fortu.modules.plotting_utils import plot_heatmap
import torch
import re

def main(
    model_name: str = "allenai/OLMo-1B",
    revision: str = None,  # Checkpoint revision (e.g., step100000-tokens419B)
    max_layer_idx: int = 16,  # OLMo-1B has 16 layers
    lens_path: str = None,
    single_lens: int = None,
    n_cycles: int = 0,
    use_bfloat16: bool = False,
    seed: int = 42,
    batch_size: int = 1,
    max_length: int = 256,
    max_new_tokens: int = 100,
    no_head_analysis: bool = False,
    n_samples: int = 5000
):
    """Run attention contrast analysis on OLMo model"""
    
    print(f"🔬 OLMo Attention Analysis Pipeline")
    print(f"=" * 80)
    print(f"Model: {model_name}")
    if revision:
        print(f"Checkpoint: {revision}")
    else:
        print(f"Checkpoint: main (final)")
    print(f"Max layers: {max_layer_idx}")
    print(f"Target layer: {single_lens if single_lens is not None else 'ALL'}")
    print(f"N cycles: {n_cycles}")
    print(f"Samples: {n_samples}")
    print(f"=" * 80)
    
    # Load model and tokenizer
    print("\n[1/6] Loading model...")
    # Note: OLMo requires trust_remote_code, but load_model_and_tokenizer doesn't support it
    # We need to load it directly
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    if use_bfloat16:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16, 
            trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, trust_remote_code=True,
            device_map="auto", low_cpu_mem_usage=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.eval()
    device = get_device()
    # Don't manually move to device when using device_map="auto"
    # model.to(device)
    
    # Create hooked model with OLMo-specific layer path
    if single_lens is not None:
        # OLMo uses 'model.layers.X' pattern
        layer_path = f"model.layers.{single_lens}"
        print(f"  Hooking layer: {layer_path}")
    else:
        layer_path = None
        print(f"  Hooking all layers")
    
    hooked_model = HookedModel(model, layer=layer_path)
    print(f"  ✓ Loaded with {len(hooked_model.hooks)} attention hooks")
    
    # Load lens if provided
    lens = None
    if lens_path is not None:
        print("\n[2/6] Loading lenses...")
        lens_paths = list(Path(lens_path).glob("*.pth"))
        if not lens_paths:
            raise ValueError(f"No lens found in {lens_path}")
        
        lens_paths.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))
        
        if single_lens is not None:
            lens_paths = [lens_paths[single_lens]]
            print(f"  Using single lens: {lens_paths[0]}")
        else:
            assert len(lens_paths) == max_layer_idx, \
                f"Expected {max_layer_idx} lenses, got {len(lens_paths)}"
            print(f"  Loaded {len(lens_paths)} lenses")
        
        lens = {
            int(re.search(r'\d+', p.stem).group()): 
            torch.load(p, weights_only=False, map_location=device) 
            for p in lens_paths
        }
        lens = {
            k: MultiHeadLens.from_dict(v) if isinstance(v, dict) else v 
            for k, v in lens.items()
        }
    else:
        print("\n[2/6] No lenses provided - using raw attention")
    
    # Load and pretokenize data
    print("\n[3/6] Loading dataset...")
    texts = load_text_dataset(seed=seed, n_samples=n_samples)
    print(f"  ✓ Loaded {len(texts)} text samples")
    
    # Run analysis
    print("\n[4/6] Running contrast analysis...")
    layers = list(range(max_layer_idx)) if single_lens is None else [single_lens]
    
    # Extract contrasts (returns: heatmap, cy_count, icl_heatmap, icl_cy_count, 
    #                           no_cycle_icl_heatmap, no_cycle_cy_count, d_idx, r_idx, no_cycle_idx)
    results = extract_contrasts(
        texts, hooked_model, tokenizer, 
        lens=lens, 
        n_cycles=n_cycles, 
        batch_size=batch_size, 
        max_length=max_length, 
        max_new_tokens=max_new_tokens, 
        no_head_analysis=no_head_analysis, 
        layers=layers
    )
    
    heatmap, cy_count, icl_heatmap, icl_cy_count, no_cycle_icl_heatmap, no_cycle_cy_count, d_idx, r_idx, no_cycle_idx = results
    
    # Print results
    print("\n[5/6] Analysis Results")
    print(f"=" * 80)
    print(f"Max new tokens: {max_new_tokens}")
    
    if single_lens is not None:
        print(f"\n📊 Layer {single_lens} Results:")
        
        # Natural repetition
        print(f"\n  Natural Repetition:")
        print(f"    Cycle count: {cy_count}")
        if heatmap[single_lens] is not None and isinstance(heatmap[single_lens], dict):
            heatmap_mean = heatmap[single_lens]['mean']
            heatmap_var = heatmap[single_lens]['var']
            print(f"    Heatmap (mean): {np.array2string(heatmap_mean, separator=',', max_line_width=np.inf)}")
            print(f"    Heatmap (var):  {np.array2string(heatmap_var, separator=',', max_line_width=np.inf)}")
        else:
            print(f"    Heatmap: None")
        
        # ICL repetition
        print(f"\n  ICL Repetition:")
        print(f"    Cycle count: {icl_cy_count}")
        if icl_heatmap[single_lens] is not None and isinstance(icl_heatmap[single_lens], dict):
            icl_heatmap_mean = icl_heatmap[single_lens]['mean']
            icl_heatmap_var = icl_heatmap[single_lens]['var']
            print(f"    Heatmap (mean): {np.array2string(icl_heatmap_mean, separator=',', max_line_width=np.inf)}")
            print(f"    Heatmap (var):  {np.array2string(icl_heatmap_var, separator=',', max_line_width=np.inf)}")
        else:
            print(f"    Heatmap: None")
        
        # No-cycle ICL
        print(f"\n  No-Cycle ICL:")
        print(f"    Cycle count: {no_cycle_cy_count}")
        if no_cycle_icl_heatmap[single_lens] is not None and isinstance(no_cycle_icl_heatmap[single_lens], dict):
            no_cycle_icl_heatmap_mean = no_cycle_icl_heatmap[single_lens]['mean']
            no_cycle_icl_heatmap_var = no_cycle_icl_heatmap[single_lens]['var']
            print(f"    Heatmap (mean): {np.array2string(no_cycle_icl_heatmap_mean, separator=',', max_line_width=np.inf)}")
            print(f"    Heatmap (var):  {np.array2string(no_cycle_icl_heatmap_var, separator=',', max_line_width=np.inf)}")
        else:
            print(f"    Heatmap: None")
        
        # Indices - format for alluvial plot parsing
        print(f"\n  Datapoint Indices:")
        print(f"layer {single_lens} data index: {d_idx}")
        print(f"layer {single_lens} repetition index: {r_idx}")
        print(f"layer {single_lens} no-cycle icl index: {no_cycle_idx}")
        print(f"layer {single_lens} no-cycle icl cycle count: {no_cycle_cy_count}")
    
    # Plot if full analysis or final layer
    if single_lens is None or single_lens == max_layer_idx - 1:
        print("\n[6/6] Creating visualizations...")
        
        # Convert heatmaps to plottable format (extract means)
        heatmap_for_plot = {}
        for layer_key, layer_data in heatmap.items():
            if layer_data is not None and isinstance(layer_data, dict):
                heatmap_for_plot[layer_key] = layer_data['mean']
            else:
                heatmap_for_plot[layer_key] = layer_data
        
        plot_heatmap(
            heatmap_for_plot,
            title=f"OLMo Unexpected Activation Contrast (n_cycles={n_cycles})",
            subtitle=f"Proportion of cycles: {cy_count:.3f}",
            xlabel="Attention Heads",
            ylabel="Layers",
            save_path=f"olmo_unexpected_heatmap_cyc{n_cycles}.png"
        )
        print(f"  ✓ Saved: olmo_unexpected_heatmap_cyc{n_cycles}.png")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Analysis complete!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    typer.run(main)
