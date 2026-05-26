import typer
import numpy as np
from pathlib import Path
from parrots.aa_fortu.modules.model_utils import HookedModel, load_model_and_tokenizer, get_device
from parrots.aa_fortu.aa_fortu_train_lens import Lens # this is what's loaded in the lens - we need it even if we don't call it
from parrots.aa_fortu.aa_fortu_train_multihead_lens import MultiHeadLens
from parrots.aa_fortu.modules.data_utils import load_text_dataset, pretokenize_texts
from parrots.aa_fortu.modules.contrast_analysis import extract_contrasts
from parrots.aa_fortu.modules.plotting_utils import plot_heatmap
import torch
import re

def main(
    model_name: str = "EleutherAI/pythia-1.4b",
    revision: str = None,
    max_layer_idx: int = 24,
    lens_path=None,
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
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, revision, use_bfloat16)
    model.eval()
    device = get_device()
    model.to(device)
    hooked_model = HookedModel(model, layer=None if single_lens is None else f"gpt_neox.layers.{str(single_lens)}")
    print(f"Loaded model {model_name} with {len(hooked_model.hooks)} attention hooks")
    # Load lens if provided
    if lens_path is not None:
        lens_paths = list(Path(lens_path).glob("*.pth"))
        if not lens_paths:
            raise ValueError(f"No lens found in {lens_path}")
        lens_paths.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))
        if single_lens is not None:
            lens_paths = [lens_paths[single_lens]]
            print(f"Keeping only lens {lens_paths[0]}")
        else:
            assert len(lens_paths) == max_layer_idx, f"Expected {max_layer_idx} lenses, one for each layer, instead got {len(lens_paths)}"
        # lens = torch.load(lens_paths[single_lens if single_lens is None else 0], weights_only=False, map_location=device)
        lens = {int(re.search(r'\d+', p.stem).group()): torch.load(p, weights_only=False, map_location=device) for p in lens_paths}
        lens={k: MultiHeadLens.from_dict(v) if isinstance(v, dict) else v for k, v in lens.items()}
    else:
        lens = None
    # Load and pretokenize data
    texts = load_text_dataset(seed=seed, n_samples=n_samples)
    # Run analysis
    layers = list(range(max_layer_idx)) if single_lens is None else [single_lens]
    
    # Updated to handle the new return format with variance
    heatmap, cy_count, icl_heatmap, icl_cy_count, no_cycle_icl_heatmap, no_cycle_cy_count, d_idx, r_idx, no_cycle_idx = extract_contrasts(
        texts, hooked_model, tokenizer, lens=lens, n_cycles=n_cycles, batch_size=batch_size, max_length=max_length, max_new_tokens=max_new_tokens, no_head_analysis=no_head_analysis, layers=layers)
    
    # Plot and save heatmap
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"layer {single_lens} cycle count: {cy_count}")
    
    # Handle the new dictionary format with 'mean' and 'var' keys
    if heatmap[single_lens] is not None and isinstance(heatmap[single_lens], dict):
        heatmap_mean = heatmap[single_lens]['mean']
        heatmap_var = heatmap[single_lens]['var']
        print(f"layer {single_lens} natural heatmap: {np.array2string(heatmap_mean, separator=',', max_line_width=np.inf)}")
        print(f"layer {single_lens} natural heatmap (variance): {np.array2string(heatmap_var, separator=',', max_line_width=np.inf)}")
    else:
        print(f"layer {single_lens} natural heatmap: None")
    
    print(f"layer {single_lens} icl cycle count: {icl_cy_count}")
    
    # Handle ICL heatmap
    if icl_heatmap[single_lens] is not None and isinstance(icl_heatmap[single_lens], dict):
        icl_heatmap_mean = icl_heatmap[single_lens]['mean']
        icl_heatmap_var = icl_heatmap[single_lens]['var']
        print(f"layer {single_lens} icl heatmap: {np.array2string(icl_heatmap_mean, separator=',', max_line_width=np.inf)}")
        print(f"layer {single_lens} icl heatmap (variance): {np.array2string(icl_heatmap_var, separator=',', max_line_width=np.inf)}")
    else:
        print(f"layer {single_lens} icl heatmap: None")
    
    # Handle no-cycle ICL heatmap
    print(f"layer {single_lens} no-cycle icl cycle count: {no_cycle_cy_count}")
    if no_cycle_icl_heatmap[single_lens] is not None and isinstance(no_cycle_icl_heatmap[single_lens], dict):
        no_cycle_icl_heatmap_mean = no_cycle_icl_heatmap[single_lens]['mean']
        no_cycle_icl_heatmap_var = no_cycle_icl_heatmap[single_lens]['var']
        print(f"layer {single_lens} no-cycle icl heatmap: {np.array2string(no_cycle_icl_heatmap_mean, separator=',', max_line_width=np.inf)}")
        print(f"layer {single_lens} no-cycle icl heatmap (variance): {np.array2string(no_cycle_icl_heatmap_var, separator=',', max_line_width=np.inf)}")
    else:
        print(f"layer {single_lens} no-cycle icl heatmap: None")
    
    print(f"layer {single_lens} data index: {d_idx}")
    print(f"layer {single_lens} repetition index: {r_idx}")
    print(f"layer {single_lens} no-cycle index: {no_cycle_idx}")
    
    if single_lens is None or single_lens == max_layer_idx - 1:
        # For plotting, use the mean values
        heatmap_for_plot = {}
        for layer_key, layer_data in heatmap.items():
            if layer_data is not None and isinstance(layer_data, dict):
                heatmap_for_plot[layer_key] = layer_data['mean']
            else:
                heatmap_for_plot[layer_key] = layer_data
        
        plot_heatmap(
            heatmap_for_plot,
            title=f"unexpected activation contrast Heatmap {n_cycles}",
            subtitle=f"proportion of cycles: {cy_count}, cycle number: {n_cycles}",
            xlabel="Attention Heads",
            ylabel="Samples",
            save_path=f"unexpected_lens_heatmap_contrast_{n_cycles}.png"
        )

if __name__ == "__main__":
    typer.run(main)
