#!/usr/bin/env python3
"""
Dual Alluvial Plot from Cycle Evolution Data
Creates the same beautiful dual alluvial plot style but using cycle evolution JSON data
instead of requiring full multihead analysis
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from pathlib import Path
import argparse
from collections import defaultdict

class DualAlluvialFromCycleEvolution:
    """Create dual alluvial plot from cycle evolution data"""
    
    def __init__(self):
        # Same colors as run_alluvial_dual.py
        self.paper_colors = {
            "non_repeating": '#D0D0D0',           # Grey
            "repeating_since_step1": '#4A90E2',   # Blue
            "repeating_since_step1000": '#9B59B6', # Purple  
            "repeating_since_step5000": '#E67E22', # Orange
            "repeating_since_step10000": '#E74C3C', # Red
            "repeating_since_step100000": '#27AE60', # Green
            "repeating_since_steplatest": '#8B4513' # Brown
        }
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
    
    def load_cycle_evolution_data(self, results_file):
        """Load cycle evolution results"""
        with open(results_file, 'r') as f:
            data = json.load(f)
        return {cp: {int(k): v for k, v in status.items()} 
                for cp, status in data.items()}
    
    def categorize_by_first_repetition(self, all_status, checkpoints, n_texts):
        """Assign each sequence to a category based on when it first started repeating"""
        categories = {}
        
        for idx in range(n_texts):
            first_rep = None
            for cp in checkpoints:
                if cp in all_status and idx in all_status[cp]:
                    if all_status[cp][idx]:
                        first_rep = cp
                        break
            
            if first_rep is None:
                categories[idx] = 'non_repeating'
            else:
                categories[idx] = f'repeating_since_{first_rep}'
        
        return categories
    
    def compute_category_counts_at_checkpoint(self, all_status, checkpoints, categories, checkpoint_idx):
        """Count sequences in each category at a specific checkpoint"""
        cp = checkpoints[checkpoint_idx]
        counts = defaultdict(int)
        
        for idx, cat in categories.items():
            if cp in all_status and idx in all_status[cp]:
                is_repeating = all_status[cp][idx]
                
                # At each checkpoint, we only show categories that have emerged by then
                if cat == 'non_repeating':
                    counts['non_repeating'] += 1
                else:
                    # Check if this category has emerged yet
                    cat_checkpoint = cat.replace('repeating_since_', '')
                    cat_idx = checkpoints.index(cat_checkpoint) if cat_checkpoint in checkpoints else 999
                    
                    if cat_idx <= checkpoint_idx:
                        counts[cat] += 1
                    elif is_repeating:
                        # Not yet categorized but is repeating - use current checkpoint
                        counts[f'repeating_since_{cp}'] += 1
                    else:
                        counts['non_repeating'] += 1
        
        return dict(counts)
    
    def draw_curved_flow(self, ax, x0, y0, h0, x1, y1, h1, color, alpha=0.5):
        """Draw curved flow between two bars"""
        # Control points for bezier curve
        cx = (x0 + x1) / 2
        
        verts = [
            (x0, y0),           # bottom-left
            (cx, y0),           # control point
            (cx, y1),           # control point
            (x1, y1),           # bottom-right
            (x1, y1 + h1),      # top-right
            (cx, y1 + h1),      # control point
            (cx, y0 + h0),      # control point
            (x0, y0 + h0),      # top-left
            (x0, y0),           # close
        ]
        
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.LINETO,
            mpath.Path.CURVE3,
            mpath.Path.CURVE3,
            mpath.Path.LINETO,
            mpath.Path.CLOSEPOLY,
        ]
        
        path = mpath.Path(verts, codes)
        patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(patch)
    
    def create_subplot_alluvial(self, ax, all_status, checkpoints, categories, title, show_ylabel=True):
        """Create a single alluvial subplot"""
        n_texts = len(categories)
        n_checkpoints = len(checkpoints)
        
        # X positions for each checkpoint
        x_positions = np.linspace(0, 1, n_checkpoints)
        bar_width = 0.08
        
        # Get category order (non_repeating first, then by checkpoint order)
        all_cats = set(categories.values())
        cat_order = ['non_repeating']
        for cp in checkpoints:
            cat = f'repeating_since_{cp}'
            if cat in all_cats:
                cat_order.append(cat)
        
        # Store bar positions for flows
        bar_positions = {}  # checkpoint_idx -> {category -> (y_start, height)}
        
        # Draw bars at each checkpoint
        for i, cp in enumerate(checkpoints):
            x = x_positions[i]
            counts = self.compute_category_counts_at_checkpoint(all_status, checkpoints, categories, i)
            
            y_offset = 0
            bar_positions[i] = {}
            
            for cat in cat_order:
                if cat in counts and counts[cat] > 0:
                    height = counts[cat] / n_texts
                    color = self.paper_colors.get(cat, '#D0D0D0')
                    
                    rect = patches.Rectangle(
                        (x - bar_width/2, y_offset), bar_width, height,
                        facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.85
                    )
                    ax.add_patch(rect)
                    
                    bar_positions[i][cat] = (y_offset, height)
                    y_offset += height
        
        # Draw flows between checkpoints
        for i in range(n_checkpoints - 1):
            x0 = x_positions[i] + bar_width/2
            x1 = x_positions[i+1] - bar_width/2
            
            for cat in cat_order:
                if cat in bar_positions[i] and cat in bar_positions[i+1]:
                    y0, h0 = bar_positions[i][cat]
                    y1, h1 = bar_positions[i+1][cat]
                    color = self.paper_colors.get(cat, '#D0D0D0')
                    self.draw_curved_flow(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.4)
        
        # Formatting
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_positions)
        
        # Format checkpoint labels
        labels = []
        for cp in checkpoints:
            if cp == 'steplatest':
                labels.append('Latest')
            else:
                num = cp.replace('step', '')
                if len(num) >= 4:
                    labels.append(f'{int(num)//1000}K')
                else:
                    labels.append(num)
        ax.set_xticklabels(labels)
        
        ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        if show_ylabel:
            ax.set_ylabel('Proportion of Sequences', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return all_cats
    
    def create_dual_alluvial(self, results_file, model_name, output_path):
        """Create the dual alluvial plot"""
        # Load data
        all_status = self.load_cycle_evolution_data(results_file)
        
        checkpoints = list(all_status.keys())
        # Sort checkpoints properly
        def checkpoint_sort_key(cp):
            if cp == 'steplatest':
                return float('inf')
            return int(cp.replace('step', ''))
        checkpoints = sorted(checkpoints, key=checkpoint_sort_key)
        
        n_texts = max(max(status.keys()) for status in all_status.values()) + 1
        
        # Categorize sequences
        categories = self.categorize_by_first_repetition(all_status, checkpoints, n_texts)
        
        # Split data into "ICL-like" (non-repeating at latest) and "Natural" (repeating at latest)
        natural_indices = set()
        icl_indices = set()
        
        latest_cp = checkpoints[-1]
        for idx in range(n_texts):
            if latest_cp in all_status and idx in all_status[latest_cp]:
                if all_status[latest_cp][idx]:
                    natural_indices.add(idx)
                else:
                    icl_indices.add(idx)
        
        # Create filtered status dicts
        natural_status = {cp: {k: v for k, v in status.items() if k in natural_indices}
                         for cp, status in all_status.items()}
        icl_status = {cp: {k: v for k, v in status.items() if k in icl_indices}
                     for cp, status in all_status.items()}
        
        natural_categories = {k: v for k, v in categories.items() if k in natural_indices}
        icl_categories = {k: v for k, v in categories.items() if k in icl_indices}
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ICL (non-repeating at end) on left
        if icl_categories:
            icl_cats = self.create_subplot_alluvial(
                ax1, icl_status, checkpoints, icl_categories, 
                f"ICL (Non-Repeating at End)\nn={len(icl_indices)}", show_ylabel=True
            )
        else:
            ax1.text(0.5, 0.5, f'No ICL sequences\n(all {n_texts} repeat)', 
                    ha='center', va='center', fontsize=12)
            ax1.set_title("ICL (Non-Repeating)", fontsize=13, fontweight='bold')
            icl_cats = set()
        
        # Natural (repeating at end) on right
        if natural_categories:
            natural_cats = self.create_subplot_alluvial(
                ax2, natural_status, checkpoints, natural_categories,
                f"Natural (Repeating at End)\nn={len(natural_indices)}", show_ylabel=False
            )
        else:
            ax2.text(0.5, 0.5, 'No Natural sequences', ha='center', va='center', fontsize=12)
            ax2.set_title("Natural (Repeating)", fontsize=13, fontweight='bold')
            natural_cats = set()
        
        # Legend
        all_cats = icl_cats | natural_cats
        if all_cats:
            legend_elements = []
            cat_labels = {
                "non_repeating": "Non-repeating",
                "repeating_since_step1": "Since step 1", 
                "repeating_since_step1000": "Since step 1K",
                "repeating_since_step5000": "Since step 5K",
                "repeating_since_step10000": "Since step 10K", 
                "repeating_since_step100000": "Since step 100K",
                "repeating_since_steplatest": "Since latest"
            }
            
            for cat in ['non_repeating'] + [f'repeating_since_{cp}' for cp in checkpoints]:
                if cat in all_cats:
                    legend_elements.append(
                        patches.Patch(facecolor=self.paper_colors.get(cat, '#D0D0D0'), 
                                    alpha=0.85, edgecolor='white', linewidth=1,
                                    label=cat_labels.get(cat, cat))
                    )
            
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=min(len(legend_elements), 4), fontsize=10, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.95)
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Dual alluvial saved to {output_path}")
        
        # Also save PDF
        pdf_path = str(output_path).replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"📊 PDF saved to {pdf_path}")
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create dual alluvial from cycle evolution data")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Pythia")
    parser.add_argument("--output_dir", type=str, default="./alluvial_plots")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model = args.model_name.replace("/", "_")
    output_path = output_dir / f"alluvial_dual_{safe_model}.png"
    
    generator = DualAlluvialFromCycleEvolution()
    generator.create_dual_alluvial(args.results_file, args.model_name, output_path)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
