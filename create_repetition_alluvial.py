#!/usr/bin/env python3
"""
Repetition Alluvial Plot from Cycle Evolution Data
Creates beautiful alluvial diagrams showing how repetition evolves across training
without requiring multihead lenses
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
import matplotlib.path as mpath
from scipy import interpolate
from pathlib import Path
import argparse

class RepetitionAlluvial:
    """Create alluvial plot from cycle evolution data"""
    
    def __init__(self):
        # Paper-appropriate colors
        self.colors = {
            'non_repeating': '#D0D0D0',          # Grey
            'repeating_since_step1': '#4A90E2',   # Blue - original
            'repeating_since_step1000': '#9B59B6', # Purple - early learnt
            'repeating_since_step5000': '#E67E22', # Orange
            'repeating_since_step10000': '#E74C3C', # Red - late learnt
            'repeating_since_step100000': '#27AE60', # Green
            'repeating_since_steplatest': '#8B4513' # Brown
        }
        
        # Simplified color scheme
        self.simple_colors = {
            'never': '#D0D0D0',           # Grey - never repeats
            'original': '#e74c3c',        # Red - original
            'learnt_early': '#f39c12',    # Orange - learnt early
            'learnt_late': '#9b59b6',     # Purple - learnt late
        }
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
    
    def load_evolution_data(self, results_file):
        """Load cycle evolution results"""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to int
        return {cp: {int(k): v for k, v in status.items()} 
                for cp, status in data.items()}
    
    def categorize_sequences(self, all_status, checkpoints):
        """Categorize sequences by when they first started repeating"""
        n_texts = max(max(all_status[cp].keys()) for cp in checkpoints) + 1
        
        categories = {}
        first_repeating_checkpoint = {}
        
        for idx in range(n_texts):
            first_rep = None
            for cp in checkpoints:
                if cp in all_status and idx in all_status[cp]:
                    if all_status[cp][idx]:
                        first_rep = cp
                        break
            
            first_repeating_checkpoint[idx] = first_rep
            
            if first_rep is None:
                categories[idx] = 'never'
            elif first_rep == 'step1':
                categories[idx] = 'original'
            elif first_rep in ['step1000', 'step5000']:
                categories[idx] = 'learnt_early'
            else:
                categories[idx] = 'learnt_late'
        
        return categories, first_repeating_checkpoint
    
    def compute_flows(self, all_status, checkpoints, categories):
        """Compute flow data for alluvial plot"""
        n_texts = len(categories)
        
        # Count transitions between checkpoints
        flows = []
        
        for i in range(len(checkpoints) - 1):
            cp1, cp2 = checkpoints[i], checkpoints[i+1]
            
            # Track flows between states
            flow_counts = {}
            
            for idx, cat in categories.items():
                if cp1 in all_status and cp2 in all_status:
                    if idx in all_status[cp1] and idx in all_status[cp2]:
                        state1 = 'repeating' if all_status[cp1][idx] else 'non_repeating'
                        state2 = 'repeating' if all_status[cp2][idx] else 'non_repeating'
                        
                        key = (state1, state2, cat)
                        flow_counts[key] = flow_counts.get(key, 0) + 1
            
            flows.append(flow_counts)
        
        return flows
    
    def draw_flow(self, ax, x0, y0, x1, y1, width, color, alpha=0.5):
        """Draw a curved flow between two points"""
        verts = [
            (x0, y0),
            (x0 + (x1-x0)*0.3, y0),
            (x0 + (x1-x0)*0.7, y1),
            (x1, y1),
            (x1, y1 + width),
            (x0 + (x1-x0)*0.7, y1 + width),
            (x0 + (x1-x0)*0.3, y0 + width),
            (x0, y0 + width),
            (x0, y0),
        ]
        
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.LINETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CLOSEPOLY,
        ]
        
        path = mpath.Path(verts, codes)
        patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(patch)
    
    def create_alluvial(self, all_status, checkpoints, model_name, output_path):
        """Create the alluvial visualization"""
        categories, first_rep = self.categorize_sequences(all_status, checkpoints)
        n_texts = len(categories)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate positions
        n_checkpoints = len(checkpoints)
        x_positions = np.linspace(0, 1, n_checkpoints)
        bar_width = 0.06
        
        # Count states at each checkpoint by category
        checkpoint_data = {}
        for cp in checkpoints:
            checkpoint_data[cp] = {
                'original_rep': 0, 'original_nonrep': 0,
                'learnt_early_rep': 0, 'learnt_early_nonrep': 0,
                'learnt_late_rep': 0, 'learnt_late_nonrep': 0,
                'never': 0
            }
            
            for idx, cat in categories.items():
                if cp in all_status and idx in all_status[cp]:
                    is_rep = all_status[cp][idx]
                    
                    if cat == 'never':
                        checkpoint_data[cp]['never'] += 1
                    elif cat == 'original':
                        if is_rep:
                            checkpoint_data[cp]['original_rep'] += 1
                        else:
                            checkpoint_data[cp]['original_nonrep'] += 1
                    elif cat == 'learnt_early':
                        if is_rep:
                            checkpoint_data[cp]['learnt_early_rep'] += 1
                        else:
                            checkpoint_data[cp]['learnt_early_nonrep'] += 1
                    elif cat == 'learnt_late':
                        if is_rep:
                            checkpoint_data[cp]['learnt_late_rep'] += 1
                        else:
                            checkpoint_data[cp]['learnt_late_nonrep'] += 1
        
        # Draw stacked bars at each checkpoint
        for i, cp in enumerate(checkpoints):
            x = x_positions[i]
            data = checkpoint_data[cp]
            
            # Order: never, original_nonrep, original_rep, learnt_early_*, learnt_late_*
            y_offset = 0
            
            # Draw never (grey)
            if data['never'] > 0:
                height = data['never'] / n_texts
                rect = patches.Rectangle((x - bar_width/2, y_offset), bar_width, height,
                                         facecolor=self.simple_colors['never'], 
                                         edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                y_offset += height
            
            # Draw original repeating (red)
            if data['original_rep'] > 0:
                height = data['original_rep'] / n_texts
                rect = patches.Rectangle((x - bar_width/2, y_offset), bar_width, height,
                                         facecolor=self.simple_colors['original'],
                                         edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                y_offset += height
            
            # Draw learnt early repeating (orange)
            if data['learnt_early_rep'] > 0:
                height = data['learnt_early_rep'] / n_texts
                rect = patches.Rectangle((x - bar_width/2, y_offset), bar_width, height,
                                         facecolor=self.simple_colors['learnt_early'],
                                         edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                y_offset += height
            
            # Draw learnt late repeating (purple)
            if data['learnt_late_rep'] > 0:
                height = data['learnt_late_rep'] / n_texts
                rect = patches.Rectangle((x - bar_width/2, y_offset), bar_width, height,
                                         facecolor=self.simple_colors['learnt_late'],
                                         edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                y_offset += height
        
        # Add checkpoint labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Proportion of Sequences', fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Checkpoint', fontsize=12, fontweight='bold')
        ax.set_title(f'Repetition Evolution During Training\n{model_name}', 
                    fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            patches.Patch(facecolor=self.simple_colors['never'], edgecolor='black',
                         label='Never Repeats'),
            patches.Patch(facecolor=self.simple_colors['original'], edgecolor='black',
                         label='Original (repeating since step1)'),
            patches.Patch(facecolor=self.simple_colors['learnt_early'], edgecolor='black',
                         label='Learnt Early (step1000-5000)'),
            patches.Patch(facecolor=self.simple_colors['learnt_late'], edgecolor='black',
                         label='Learnt Late (step10000+)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Alluvial plot saved to {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create repetition alluvial from cycle evolution data")
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to cycle_evolution_status_*.json file")
    parser.add_argument("--model_name", type=str, default="Pythia")
    parser.add_argument("--output_dir", type=str, default="./alluvial_plots")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    alluvial = RepetitionAlluvial()
    
    # Load data
    print(f"📂 Loading data from {args.results_file}")
    all_status = alluvial.load_evolution_data(args.results_file)
    
    checkpoints = list(all_status.keys())
    # Sort checkpoints properly
    def checkpoint_sort_key(cp):
        if cp == 'steplatest':
            return float('inf')
        return int(cp.replace('step', ''))
    checkpoints = sorted(checkpoints, key=checkpoint_sort_key)
    
    print(f"📊 Checkpoints: {checkpoints}")
    
    # Create alluvial
    safe_model = args.model_name.replace("/", "_")
    output_path = output_dir / f"repetition_alluvial_{safe_model}.png"
    
    alluvial.create_alluvial(all_status, checkpoints, args.model_name, output_path)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
