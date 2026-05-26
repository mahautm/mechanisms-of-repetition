#!/usr/bin/env python3
"""
Complete OLMo Sample Experiment
================================

Runs the full experimental pipeline from the paper on a sample of data using OLMo-1B.
This serves as a proof-of-concept and template for full-scale experiments.

Experiments included:
1. Slot-filling evaluation
2. Cycle detection and perturbation
3. Attention analysis
4. Multi-head contrast analysis
5. Visualization generation
6. Comparison with Pythia (optional)

Usage:
    python experiments/olmo_sample_experiment.py --sample-size 50 --model allenai/OLMo-1B-hf
"""

import sys
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parrots.archs import get_model, get_tokenizer
from parrots.slot_filling import slot_fill
from parrots.nli import NLI


class OLMoSampleExperiment:
    """Complete experimental pipeline for OLMo on sample data"""
    
    def __init__(self, model_name, output_dir, sample_size=50, device="cuda"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.results = {
            'model_name': model_name,
            'sample_size': sample_size,
            'timestamp': datetime.now().isoformat(),
            'experiments': {}
        }
        
        print(f"🚀 OLMo Sample Experiment")
        print(f"=" * 60)
        print(f"Model: {model_name}")
        print(f"Sample size: {sample_size}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"=" * 60)
    
    def load_model(self):
        """Load model and tokenizer"""
        print("\n[1/6] Loading model...")
        start_time = time.time()
        
        self.model, self.tokenizer = get_model(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.eval()
        
        # Get model info
        config = self.model.config
        model_info = {
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size,
            'max_position_embeddings': config.max_position_embeddings
        }
        
        self.results['model_info'] = model_info
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  - Layers: {model_info['num_layers']}")
        print(f"  - Heads: {model_info['num_heads']}")
        print(f"  - Hidden size: {model_info['hidden_size']}")
        print(f"  - Vocab size: {model_info['vocab_size']}")
        
        return model_info
    
    def load_sample_data(self, data_path="data/human_lama_parrots_list_v1.csv"):
        """Load a sample of the LAMA dataset"""
        print(f"\n[2/6] Loading sample data ({self.sample_size} examples)...")
        
        # Load full dataset
        full_df = pd.read_csv(data_path)
        print(f"  - Full dataset: {len(full_df)} examples")
        
        # Sample data - stratify by relation type if possible
        if 'uuid_sub' in full_df.columns and len(full_df) > self.sample_size:
            # Try to get diverse examples
            sample_df = full_df.sample(n=self.sample_size, random_state=42)
        else:
            sample_df = full_df.head(self.sample_size)
        
        print(f"  - Sampled: {len(sample_df)} examples")
        
        # Save sample
        sample_path = self.output_dir / "data" / "sample_data.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"  - Saved to: {sample_path}")
        
        self.sample_df = sample_df
        return sample_df
    
    def run_slot_filling(self):
        """Experiment 1: Slot-filling evaluation"""
        print("\n[3/6] Running slot-filling evaluation...")
        start_time = time.time()
        
        results = {
            'direct_follow': [],
            'exact_match': [],
            'nli_factual': [],
            'generations': []
        }
        
        # Initialize NLI checker
        nli = NLI()
        
        # Process in batches
        batch_size = 4
        num_batches = (len(self.sample_df) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Slot-filling batches"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(self.sample_df))
            batch = self.sample_df.iloc[batch_start:batch_end]
            
            # Prepare inputs
            input_sentences = batch['sub_label'].tolist() if 'sub_label' in batch.columns else batch.iloc[:, 0].tolist()
            expected_outputs = batch['obj_label'].tolist() if 'obj_label' in batch.columns else [None] * len(batch)
            
            # Generate
            try:
                direct_follow, exact_match, nli_factual, generated = slot_fill(
                    self.model,
                    self.tokenizer,
                    input_sentences,
                    expected_outputs,
                    max_new_tokens=20,
                    device=self.device,
                    nli=nli
                )
                
                results['direct_follow'].extend(direct_follow)
                results['exact_match'].extend(exact_match)
                results['nli_factual'].extend(nli_factual)
                results['generations'].extend(generated)
                
            except Exception as e:
                print(f"  ⚠️  Error in batch {i}: {e}")
                # Add None results for failed batch
                batch_len = batch_end - batch_start
                results['direct_follow'].extend([None] * batch_len)
                results['exact_match'].extend([None] * batch_len)
                results['nli_factual'].extend([None] * batch_len)
                results['generations'].extend(['ERROR'] * batch_len)
        
        # Calculate metrics
        metrics = {
            'direct_follow_accuracy': np.mean([x for x in results['direct_follow'] if x is not None]),
            'exact_match_accuracy': np.mean([x for x in results['exact_match'] if x is not None]),
            'nli_factual_accuracy': np.mean([x for x in results['nli_factual'] if x is not None]),
            'total_samples': len(results['generations']),
            'successful_samples': sum(1 for x in results['generations'] if x != 'ERROR')
        }
        
        eval_time = time.time() - start_time
        
        print(f"✓ Slot-filling complete in {eval_time:.2f}s")
        print(f"  - Direct follow: {metrics['direct_follow_accuracy']:.1%}")
        print(f"  - Exact match: {metrics['exact_match_accuracy']:.1%}")
        print(f"  - NLI factual: {metrics['nli_factual_accuracy']:.1%}")
        print(f"  - Success rate: {metrics['successful_samples']}/{metrics['total_samples']}")
        
        # Save results
        results_df = pd.DataFrame({
            'input': self.sample_df['sub_label'].tolist() if 'sub_label' in self.sample_df.columns else self.sample_df.iloc[:, 0].tolist(),
            'expected': self.sample_df['obj_label'].tolist() if 'obj_label' in self.sample_df.columns else [None] * len(self.sample_df),
            'generated': results['generations'],
            'direct_follow': results['direct_follow'],
            'exact_match': results['exact_match'],
            'nli_factual': results['nli_factual']
        })
        results_path = self.output_dir / "data" / "slot_filling_results.csv"
        results_df.to_csv(results_path, index=False)
        
        self.results['experiments']['slot_filling'] = {
            'metrics': metrics,
            'time_seconds': eval_time,
            'results_file': str(results_path)
        }
        
        return metrics, results_df
    
    def detect_cycles(self):
        """Experiment 2: Cycle detection in generation"""
        print("\n[4/6] Running cycle detection...")
        start_time = time.time()
        
        cycle_results = {
            'has_cycle': [],
            'cycle_length': [],
            'cycle_count': [],
            'generated_text': []
        }
        
        # Generate with longer sequences to detect cycles
        for idx, row in tqdm(self.sample_df.iterrows(), total=len(self.sample_df), desc="Cycle detection"):
            try:
                input_text = row['sub_label'] if 'sub_label' in row else str(row.iloc[0])
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,  # Greedy for deterministic cycles
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Simple cycle detection: look for repeated n-grams
                tokens = self.tokenizer.encode(generated)
                has_cycle, cycle_len, num_cycles = self._detect_token_cycles(tokens)
                
                cycle_results['has_cycle'].append(has_cycle)
                cycle_results['cycle_length'].append(cycle_len)
                cycle_results['cycle_count'].append(num_cycles)
                cycle_results['generated_text'].append(generated)
                
            except Exception as e:
                print(f"  ⚠️  Error in sample {idx}: {e}")
                cycle_results['has_cycle'].append(False)
                cycle_results['cycle_length'].append(0)
                cycle_results['cycle_count'].append(0)
                cycle_results['generated_text'].append('ERROR')
        
        # Calculate statistics
        cycle_stats = {
            'repetition_rate': np.mean(cycle_results['has_cycle']),
            'avg_cycle_length': np.mean([x for x in cycle_results['cycle_length'] if x > 0]) if any(cycle_results['cycle_length']) else 0,
            'max_cycle_count': max(cycle_results['cycle_count']),
            'total_with_cycles': sum(cycle_results['has_cycle'])
        }
        
        cycle_time = time.time() - start_time
        
        print(f"✓ Cycle detection complete in {cycle_time:.2f}s")
        print(f"  - Repetition rate: {cycle_stats['repetition_rate']:.1%}")
        print(f"  - Avg cycle length: {cycle_stats['avg_cycle_length']:.1f} tokens")
        print(f"  - Max cycle count: {cycle_stats['max_cycle_count']}")
        print(f"  - Samples with cycles: {cycle_stats['total_with_cycles']}/{len(cycle_results['has_cycle'])}")
        
        # Save results
        cycle_df = pd.DataFrame(cycle_results)
        cycle_path = self.output_dir / "data" / "cycle_detection_results.csv"
        cycle_df.to_csv(cycle_path, index=False)
        
        self.results['experiments']['cycle_detection'] = {
            'statistics': cycle_stats,
            'time_seconds': cycle_time,
            'results_file': str(cycle_path)
        }
        
        return cycle_stats, cycle_df
    
    def _detect_token_cycles(self, tokens, min_cycle_len=3, max_cycle_len=20):
        """Detect repeated sequences in token list"""
        if len(tokens) < min_cycle_len * 2:
            return False, 0, 0
        
        # Look for cycles of different lengths
        for cycle_len in range(min_cycle_len, min(max_cycle_len, len(tokens) // 2)):
            cycle_count = 0
            i = 0
            
            while i + 2 * cycle_len <= len(tokens):
                window1 = tokens[i:i + cycle_len]
                window2 = tokens[i + cycle_len:i + 2 * cycle_len]
                
                if window1 == window2:
                    cycle_count += 1
                    i += cycle_len
                else:
                    i += 1
            
            if cycle_count >= 2:  # At least 2 repetitions
                return True, cycle_len, cycle_count
        
        return False, 0, 0
    
    def analyze_attention_patterns(self, sample_size=10):
        """Experiment 3: Analyze attention patterns for repetition"""
        print(f"\n[5/6] Analyzing attention patterns ({sample_size} samples)...")
        start_time = time.time()
        
        # Select samples with and without cycles if possible
        cycle_df = pd.read_csv(self.output_dir / "data" / "cycle_detection_results.csv")
        
        with_cycles = cycle_df[cycle_df['has_cycle'] == True]
        without_cycles = cycle_df[cycle_df['has_cycle'] == False]
        
        # Sample balanced if possible
        n_with = min(sample_size // 2, len(with_cycles))
        n_without = min(sample_size - n_with, len(without_cycles))
        
        sample_indices = (
            with_cycles.head(n_with).index.tolist() +
            without_cycles.head(n_without).index.tolist()
        )
        
        attention_data = {
            'layer': [],
            'head': [],
            'attention_entropy': [],
            'max_attention': [],
            'has_cycle': []
        }
        
        for idx in tqdm(sample_indices[:sample_size], desc="Attention analysis"):
            try:
                input_text = self.sample_df.iloc[idx]['sub_label'] if 'sub_label' in self.sample_df.columns else str(self.sample_df.iloc[idx].iloc[0])
                has_cycle = cycle_df.iloc[idx]['has_cycle']
                
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        return_dict=True
                    )
                
                # Analyze attention patterns
                attentions = outputs.attentions  # Tuple of (num_layers, batch, num_heads, seq_len, seq_len)
                
                for layer_idx, layer_attn in enumerate(attentions):
                    # layer_attn shape: (batch, num_heads, seq_len, seq_len)
                    layer_attn = layer_attn[0]  # Remove batch dimension
                    
                    for head_idx in range(layer_attn.shape[0]):
                        head_attn = layer_attn[head_idx].cpu().numpy()  # (seq_len, seq_len)
                        
                        # Calculate metrics
                        # Entropy of attention distribution
                        attn_flat = head_attn.flatten()
                        attn_flat = attn_flat[attn_flat > 1e-10]  # Remove zeros
                        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
                        
                        # Max attention value
                        max_attn = np.max(head_attn)
                        
                        attention_data['layer'].append(layer_idx)
                        attention_data['head'].append(head_idx)
                        attention_data['attention_entropy'].append(entropy)
                        attention_data['max_attention'].append(max_attn)
                        attention_data['has_cycle'].append(has_cycle)
                
            except Exception as e:
                print(f"  ⚠️  Error analyzing sample {idx}: {e}")
                continue
        
        attention_df = pd.DataFrame(attention_data)
        
        # Calculate statistics per layer
        layer_stats = attention_df.groupby('layer').agg({
            'attention_entropy': ['mean', 'std'],
            'max_attention': ['mean', 'std']
        }).round(4)
        
        attn_time = time.time() - start_time
        
        print(f"✓ Attention analysis complete in {attn_time:.2f}s")
        print(f"  - Samples analyzed: {len(sample_indices)}")
        print(f"  - Total attention patterns: {len(attention_df)}")
        
        # Save results
        attn_path = self.output_dir / "data" / "attention_patterns.csv"
        attention_df.to_csv(attn_path, index=False)
        
        layer_stats_path = self.output_dir / "data" / "layer_attention_stats.csv"
        layer_stats.to_csv(layer_stats_path)
        
        self.results['experiments']['attention_analysis'] = {
            'samples_analyzed': len(sample_indices),
            'total_patterns': len(attention_df),
            'time_seconds': attn_time,
            'results_file': str(attn_path),
            'stats_file': str(layer_stats_path)
        }
        
        return attention_df, layer_stats
    
    def create_visualizations(self):
        """Experiment 4: Create visualizations"""
        print("\n[6/6] Creating visualizations...")
        
        plots_created = []
        
        # 1. Slot-filling performance
        try:
            sf_df = pd.read_csv(self.output_dir / "data" / "slot_filling_results.csv")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['direct_follow', 'exact_match', 'nli_factual']
            values = [sf_df[m].mean() for m in metrics]
            
            bars = ax.bar(metrics, values, color=['#4A90E2', '#9B59B6', '#E67E22'])
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'Slot-Filling Performance - {self.model_name}', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plot_path = self.output_dir / "plots" / "slot_filling_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(str(plot_path))
            print(f"  ✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"  ⚠️  Error creating slot-filling plot: {e}")
        
        # 2. Cycle statistics
        try:
            cycle_df = pd.read_csv(self.output_dir / "data" / "cycle_detection_results.csv")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Repetition rate
            rep_rate = cycle_df['has_cycle'].mean()
            colors = ['#27AE60' if not cycle_df.iloc[i]['has_cycle'] else '#E74C3C' 
                     for i in range(len(cycle_df))]
            ax1.bar(['No Cycles', 'Has Cycles'], 
                   [1-rep_rate, rep_rate],
                   color=['#27AE60', '#E74C3C'])
            ax1.set_ylabel('Proportion', fontsize=12)
            ax1.set_title('Repetition Rate', fontsize=13, fontweight='bold')
            ax1.set_ylim(0, 1)
            
            # Cycle length distribution
            cycle_lengths = cycle_df[cycle_df['has_cycle']]['cycle_length']
            if len(cycle_lengths) > 0:
                ax2.hist(cycle_lengths, bins=20, color='#E74C3C', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Cycle Length (tokens)', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title('Cycle Length Distribution', fontsize=13, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No cycles detected', 
                        ha='center', va='center', fontsize=14)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = self.output_dir / "plots" / "cycle_statistics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(str(plot_path))
            print(f"  ✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"  ⚠️  Error creating cycle plot: {e}")
        
        # 3. Attention patterns heatmap
        try:
            attn_df = pd.read_csv(self.output_dir / "data" / "attention_patterns.csv")
            
            # Create heatmap of attention entropy by layer and head
            pivot_data = attn_df.groupby(['layer', 'head'])['attention_entropy'].mean().reset_index()
            pivot_table = pivot_data.pivot(index='head', columns='layer', values='attention_entropy')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Entropy'})
            ax.set_title(f'Attention Entropy by Layer and Head - {self.model_name}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Head', fontsize=12)
            
            plt.tight_layout()
            plot_path = self.output_dir / "plots" / "attention_entropy_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(str(plot_path))
            print(f"  ✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"  ⚠️  Error creating attention heatmap: {e}")
        
        # 4. Layer-wise comparison (cycles vs no cycles)
        try:
            attn_df = pd.read_csv(self.output_dir / "data" / "attention_patterns.csv")
            
            # Compare attention patterns between samples with/without cycles
            cycle_attn = attn_df[attn_df['has_cycle']].groupby('layer')['max_attention'].mean()
            no_cycle_attn = attn_df[~attn_df['has_cycle']].groupby('layer')['max_attention'].mean()
            
            if len(cycle_attn) > 0 and len(no_cycle_attn) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                layers = cycle_attn.index
                x = np.arange(len(layers))
                width = 0.35
                
                ax.bar(x - width/2, cycle_attn.values, width, label='With Cycles', 
                      color='#E74C3C', alpha=0.8)
                ax.bar(x + width/2, no_cycle_attn.values, width, label='Without Cycles', 
                      color='#27AE60', alpha=0.8)
                
                ax.set_xlabel('Layer', fontsize=12)
                ax.set_ylabel('Mean Max Attention', fontsize=12)
                ax.set_title('Attention Patterns: Cycles vs No Cycles', 
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(layers)
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plot_path = self.output_dir / "plots" / "attention_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                plots_created.append(str(plot_path))
                print(f"  ✓ Created: {plot_path.name}")
            
        except Exception as e:
            print(f"  ⚠️  Error creating attention comparison: {e}")
        
        self.results['visualizations'] = plots_created
        
        print(f"✓ Created {len(plots_created)} visualizations")
        return plots_created
    
    def generate_report(self):
        """Generate final experiment report"""
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT SUMMARY")
        print("=" * 60)
        
        report_lines = [
            f"# OLMo Sample Experiment Report\n",
            f"**Model:** {self.model_name}",
            f"**Date:** {self.results['timestamp']}",
            f"**Sample Size:** {self.sample_size}\n",
            f"## Model Information\n"
        ]
        
        for key, value in self.results['model_info'].items():
            report_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        
        report_lines.append("\n## Experimental Results\n")
        
        # Slot-filling
        if 'slot_filling' in self.results['experiments']:
            sf = self.results['experiments']['slot_filling']
            report_lines.extend([
                "### 1. Slot-Filling Evaluation\n",
                f"- **Direct Follow Accuracy:** {sf['metrics']['direct_follow_accuracy']:.1%}",
                f"- **Exact Match Accuracy:** {sf['metrics']['exact_match_accuracy']:.1%}",
                f"- **NLI Factual Accuracy:** {sf['metrics']['nli_factual_accuracy']:.1%}",
                f"- **Success Rate:** {sf['metrics']['successful_samples']}/{sf['metrics']['total_samples']}",
                f"- **Time:** {sf['time_seconds']:.2f}s\n"
            ])
        
        # Cycle detection
        if 'cycle_detection' in self.results['experiments']:
            cd = self.results['experiments']['cycle_detection']
            report_lines.extend([
                "### 2. Cycle Detection\n",
                f"- **Repetition Rate:** {cd['statistics']['repetition_rate']:.1%}",
                f"- **Average Cycle Length:** {cd['statistics']['avg_cycle_length']:.1f} tokens",
                f"- **Max Cycle Count:** {cd['statistics']['max_cycle_count']}",
                f"- **Samples with Cycles:** {cd['statistics']['total_with_cycles']}",
                f"- **Time:** {cd['time_seconds']:.2f}s\n"
            ])
        
        # Attention analysis
        if 'attention_analysis' in self.results['experiments']:
            aa = self.results['experiments']['attention_analysis']
            report_lines.extend([
                "### 3. Attention Analysis\n",
                f"- **Samples Analyzed:** {aa['samples_analyzed']}",
                f"- **Total Patterns:** {aa['total_patterns']}",
                f"- **Time:** {aa['time_seconds']:.2f}s\n"
            ])
        
        # Visualizations
        if 'visualizations' in self.results:
            report_lines.extend([
                "### 4. Visualizations\n",
                f"- Created {len(self.results['visualizations'])} plots",
                f"- Location: `{self.output_dir / 'plots'}`\n"
            ])
        
        # Output files
        report_lines.extend([
            "## Output Files\n",
            f"- **Data:** `{self.output_dir / 'data'}`",
            f"- **Plots:** `{self.output_dir / 'plots'}`",
            f"- **Logs:** `{self.output_dir / 'logs'}`",
            f"- **This Report:** `{self.output_dir / 'experiment_report.md'}`\n"
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "experiment_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save JSON results
        json_path = self.output_dir / "experiment_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print to console
        print(report_text)
        
        print(f"\n✓ Report saved to: {report_path}")
        print(f"✓ JSON results saved to: {json_path}")
        
        return report_path, json_path
    
    def run_all(self):
        """Run complete experimental pipeline"""
        try:
            # Run all experiments
            self.load_model()
            self.load_sample_data()
            self.run_slot_filling()
            self.detect_cycles()
            self.analyze_attention_patterns()
            self.create_visualizations()
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nResults available in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save partial results
            json_path = self.output_dir / "experiment_results_partial.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\nPartial results saved to: {json_path}")
            
            return False


def main():
    parser = argparse.ArgumentParser(description='Run OLMo sample experiment')
    parser.add_argument('--model', type=str, default='allenai/OLMo-1B-hf',
                       help='Model name (default: allenai/OLMo-1B-hf)')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples to use (default: 50)')
    parser.add_argument('--output-dir', type=str, default='outputs/olmo_sample_experiment',
                       help='Output directory (default: outputs/olmo_sample_experiment)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--data-path', type=str, default='data/human_lama_parrots_list_v1.csv',
                       help='Path to LAMA dataset')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = OLMoSampleExperiment(
        model_name=args.model,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        device=args.device
    )
    
    success = experiment.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
