#!/usr/bin/env python3
"""
Generate bias ratio analysis for all training steps/revisions.
This script finds all step directories and runs the full analysis pipeline for each.
"""

print("🔧 Starting imports...")
import subprocess
import sys
from pathlib import Path
import argparse
import time

print("✅ All imports successful!")

def find_all_steps(base_dir):
    """Find all step directories in the cycle evolution results."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Base directory does not exist: {base_path}")
        return []
    
    # Look for step directories (stepXXX and steplatest)
    step_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            if item.name.startswith('step') or item.name == 'steplatest':
                step_dirs.append(item)
    
    # Sort by step number (put steplatest at the end)
    def sort_key(path):
        name = path.name
        if name == 'steplatest':
            return float('inf')  # Put steplatest last
        elif name.startswith('step'):
            try:
                return int(name[4:])  # Extract number after 'step'
            except ValueError:
                return float('inf')
        else:
            return float('inf')
    
    step_dirs.sort(key=sort_key)
    
    print(f"📊 Found {len(step_dirs)} step directories:")
    for step_dir in step_dirs:
        print(f"   - {step_dir.name}")
    
    return step_dirs

def check_cycle_data_exists(step_dir):
    """Check if cycle evolution data files exist for this step."""
    pt_files = list(step_dir.glob("cycle_evolution_parametric_c*_l*_*.pt"))
    return len(pt_files) > 0

def run_prompt_attention_analysis(step_dir, output_base_dir):
    """Run the prompt vs attention analysis for a specific step."""
    step_name = step_dir.name
    output_dir = output_base_dir / f"prompt_vs_attention_{step_name}"
    
    print(f"\n🔍 Processing step: {step_name}")
    print(f"   📂 Data path: {step_dir}")
    print(f"   📁 Output path: {output_dir}")
    
    # Check if data exists
    if not check_cycle_data_exists(step_dir):
        print(f"   ⚠️  No cycle evolution data found for {step_name}, skipping...")
        return None
    
    try:
        # Run the prompt vs attention analysis
        cmd = [
            sys.executable, "analyze_prompt_vs_attention.py",
            "--data_path", str(step_dir),
            "--output_dir", str(output_dir)
        ]
        
        print(f"   🚀 Running prompt vs attention analysis...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"   ❌ Error in prompt vs attention analysis:")
            print(f"      {result.stderr}")
            return None
        
        print(f"   ✅ Prompt vs attention analysis complete")
        return output_dir
        
    except Exception as e:
        print(f"   ❌ Exception during analysis: {e}")
        return None

def run_bias_ratio_summary(reports_dir, output_dir, step_name):
    """Run the bias ratio summary analysis for a specific step."""
    
    print(f"   📊 Creating bias ratio summary for {step_name}...")
    
    try:
        # Run the bias ratio summary
        cmd = [
            sys.executable, "create_bias_ratio_summary.py",
            "--reports_dir", str(reports_dir),
            "--output_dir", str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"   ❌ Error in bias ratio summary:")
            print(f"      {result.stderr}")
            return False
        
        print(f"   ✅ Bias ratio summary complete")
        return True
        
    except Exception as e:
        print(f"   ❌ Exception during summary: {e}")
        return False

def create_cross_step_comparison(all_step_results, output_base_dir):
    """Create a comparison plot across all training steps."""
    
    print(f"\n📈 Creating cross-step comparison...")
    
    # This would be a more complex analysis comparing bias ratios across steps
    # For now, we'll create a simple summary report
    
    summary_dir = output_base_dir / "cross_step_comparison"
    summary_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a summary report
    report_lines = []
    report_lines.append("# Cross-Step Attention Bias Analysis")
    report_lines.append("")
    report_lines.append("## Processing Summary")
    report_lines.append("")
    
    for step_name, result_info in all_step_results.items():
        if result_info['success']:
            report_lines.append(f"### {step_name}")
            report_lines.append(f"- ✅ **Status**: Successfully processed")
            report_lines.append(f"- 📊 **Reports**: {result_info['reports_dir']}")
            report_lines.append(f"- 📈 **Summary**: {result_info['summary_dir']}")
            report_lines.append("")
        else:
            report_lines.append(f"### {step_name}")
            report_lines.append(f"- ❌ **Status**: Failed or skipped")
            if result_info.get('error'):
                report_lines.append(f"- **Error**: {result_info['error']}")
            report_lines.append("")
    
    report_lines.append("## Analysis Overview")
    report_lines.append("")
    report_lines.append("Each step directory contains:")
    report_lines.append("1. **Individual layer reports**: Detailed bias ratios for each layer")
    report_lines.append("2. **Summary visualizations**: Heatmaps, bar charts, and evolution plots")
    report_lines.append("3. **Statistical summaries**: Key findings and interpretations")
    report_lines.append("")
    report_lines.append("## Next Steps")
    report_lines.append("")
    report_lines.append("To compare across training steps:")
    report_lines.append("1. Compare the `attention_bias_summary.png` files across steps")
    report_lines.append("2. Look at how newline vs content word bias changes over training")
    report_lines.append("3. Examine if template word specialization emerges gradually")
    
    # Save the report
    report_path = summary_dir / "cross_step_analysis_summary.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   ✅ Cross-step summary saved: {report_path}")
    return summary_dir

def main():
    parser = argparse.ArgumentParser(description="Generate bias ratio analysis for all training steps")
    parser.add_argument("--base_dir", type=str,
                       default="/home/mmahaut/projects/parrots/cycle-attention-analysis/src/plots/cycle_evolution_parametric/cycles_4",
                       help="Base directory containing step subdirectories")
    parser.add_argument("--output_base_dir", type=str, 
                       default="./plots/multi_step_analysis",
                       help="Base output directory for all analyses")
    parser.add_argument("--steps", type=str, nargs="*", 
                       help="Specific steps to process (default: all found)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_base_dir = Path(args.output_base_dir)
    output_base_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 Multi-step bias ratio analysis starting...")
    print(f"📂 Base directory: {base_dir}")
    print(f"📁 Output base directory: {output_base_dir}")
    
    # Find all step directories
    all_step_dirs = find_all_steps(base_dir)
    
    if not all_step_dirs:
        print("❌ No step directories found!")
        return
    
    # Filter to specific steps if requested
    if args.steps:
        requested_steps = set(args.steps)
        all_step_dirs = [d for d in all_step_dirs if d.name in requested_steps]
        print(f"📊 Filtered to requested steps: {[d.name for d in all_step_dirs]}")
    
    # Process each step
    all_results = {}
    successful_count = 0
    
    start_time = time.time()
    
    for step_dir in all_step_dirs:
        step_name = step_dir.name
        
        try:
            # Step 1: Run prompt vs attention analysis
            reports_dir = run_prompt_attention_analysis(step_dir, output_base_dir)
            
            if reports_dir is None:
                all_results[step_name] = {
                    'success': False,
                    'error': 'Prompt vs attention analysis failed'
                }
                continue
            
            # Step 2: Run bias ratio summary
            summary_dir = output_base_dir / f"bias_summary_{step_name}"
            success = run_bias_ratio_summary(reports_dir, summary_dir, step_name)
            
            if success:
                all_results[step_name] = {
                    'success': True,
                    'reports_dir': reports_dir,
                    'summary_dir': summary_dir
                }
                successful_count += 1
            else:
                all_results[step_name] = {
                    'success': False,
                    'error': 'Bias ratio summary failed'
                }
                
        except Exception as e:
            print(f"   ❌ Unexpected error processing {step_name}: {e}")
            all_results[step_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Create cross-step comparison
    comparison_dir = create_cross_step_comparison(all_results, output_base_dir)
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n✅ Multi-step analysis complete!")
    print(f"📊 Successfully processed: {successful_count}/{len(all_step_dirs)} steps")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📁 Results saved to: {output_base_dir}")
    
    # Print summary of successful results
    print(f"\n📈 Successful analyses:")
    for step_name, result_info in all_results.items():
        if result_info['success']:
            print(f"   ✅ {step_name}: {result_info['summary_dir']}")
        else:
            print(f"   ❌ {step_name}: {result_info.get('error', 'Unknown error')}")
    
    print(f"\n🔍 Cross-step comparison: {comparison_dir}")

if __name__ == "__main__":
    main()