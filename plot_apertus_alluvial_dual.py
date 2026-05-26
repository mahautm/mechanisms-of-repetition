#!/usr/bin/env python3
"""
Generate Apertus-8B Dual Alluvial Plot
Reads data from log file 1841131 (step50000-tokens210B)
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

from run_alluvial_dual import BeautifulDualAlluvial
from pathlib import Path

def main():
    print("🎨 Creating Apertus-8B Alluvial Plot")
    print("=" * 80)
    
    # Configuration
    model_name = "swiss-ai/Apertus-8B-2509"
    layer = 24  # Layer 24 out of 32 (75% depth)
    log_dir = "logs"
    
    # Job ID to checkpoint mapping - all 6 checkpoints
    APERTUS_JOB_MAP = {
        "step50000-tokens210B": "1841716",
        "step500000-tokens2100B": "1841717",
        "step1000000-tokens4200B": "1841718",
        "step1700000-tokens7232B": "1841719",
        "step2300000-tokens12272B": "1841720",
        "step2627139-tokens15T": "1841721",
    }
    
    # Initialize generator
    generator = BeautifulDualAlluvial()
    
    # Load natural data from logs
    print(f"\n📂 Loading natural repetition data from {log_dir}...")
    natural_data = generator.load_natural_repetition_data_from_logs(
        log_dir=log_dir,
        job_checkpoint_map=APERTUS_JOB_MAP,
        layer=layer,
        model_name=model_name
    )
    
    if not natural_data:
        print("❌ No natural repetition data found!")
        return
    
    # Load no-cycle ICL data from logs
    print(f"\n📂 Loading no-cycle ICL data from {log_dir}...")
    no_cycle_icl_data = generator.load_no_cycle_icl_data_from_logs(
        log_dir=log_dir,
        job_checkpoint_map=APERTUS_JOB_MAP,
        layer=layer,
        model_name=model_name
    )
    
    if not no_cycle_icl_data:
        print("❌ No no-cycle ICL data found!")
        return
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Natural checkpoints: {list(natural_data.keys())}")
    print(f"  No-cycle ICL checkpoints: {list(no_cycle_icl_data.keys())}")
    
    # Create progressive categorization
    print(f"\n🔄 Creating progressive categorization...")
    
    # All 6 Apertus checkpoints in training order
    checkpoint_order = [
        "step50000-tokens210B",
        "step500000-tokens2100B", 
        "step1000000-tokens4200B",
        "step1700000-tokens7232B",
        "step2300000-tokens12272B",
        "step2627139-tokens15T"
    ]
    
    # Create categorization
    categorization, all_datapoints = generator.create_progressive_categorization(
        natural_data, 
        checkpoint_order
    )
    
    print(f"  Categories created for {len(all_datapoints)} datapoints")
    print(f"  Checkpoint-category mapping: {list(categorization.keys())}")
    
    # Create plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 Generating dual alluvial plot...")
    
    output_path = output_dir / "apertus_layer24_dual_alluvial.png"
    
    result_path = generator.create_beautiful_dual_alluvial(
        natural_data=natural_data,
        no_cycle_icl_data=no_cycle_icl_data,
        output_path=output_path,
        checkpoints=checkpoint_order,
        label_prefix=None  # Use token counts directly like "210B", "15T"
    )
    
    if result_path:
        print(f"\n✅ Plot saved to: {result_path}")
    else:
        print(f"\n❌ Failed to generate plot")
    print("=" * 80)
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"Model: {model_name}")
    print(f"Layer: {layer} (75% depth, 24/32)")
    print(f"Checkpoint: step50000-tokens210B")
    
    checkpoint = "step50000-tokens210B"
    if checkpoint in natural_data:
        nat = natural_data[checkpoint]
        print(f"\nNatural Repetition:")
        print(f"  Total samples: {len(nat['data_indices'])}")
        print(f"  Repetitions detected: {len(nat['repetition_indices'])}")
        
    if checkpoint in no_cycle_icl_data:
        icl = no_cycle_icl_data[checkpoint]
        print(f"\nNo-Cycle ICL:")
        print(f"  Total samples: {icl['cycle_count']}")
    
    # Show categorization breakdown
    print(f"\nCategorization Breakdown:")
    for category, indices in categorization.items():
        if indices:
            print(f"  {category}: {len(indices)} samples")

if __name__ == "__main__":
    main()
