#!/usr/bin/env python3
"""
Generate OLMo Alluvial Plot using existing run_alluvial_dual.py infrastructure
Just changes the input path - minimal modifications
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

from pathlib import Path
from run_alluvial_dual import BeautifulDualAlluvial

# OLMo checkpoint to job ID mapping (1000 samples)
OLMO_JOB_MAP = {
    'step1000-tokens4B': '1840088',
    'step343000-tokens1438B': '1840089',
    'step425000-tokens1781B': '1840090',
    'step509000-tokens2134B': '1840091',
    'step593000-tokens2486B': '1840092',
    'step738020-tokens3094B': '1840093',
}

def main():
    print("OLMo Alluvial Plot Generator")
    print("=" * 33)
    
    log_dir = "logs"
    layer = 12
    output_file = "plots/olmo_alluvial_layer12.png"
    
    generator = BeautifulDualAlluvial()
    
    print(f"\nProcessing OLMo-1B...")
    
    # Load natural data from logs
    print("Loading natural repetition data...")
    natural_data = generator.load_natural_repetition_data_from_logs(
        log_dir, OLMO_JOB_MAP, layer, model_name="OLMo-1B"
    )
    
    # Load no-cycle ICL data from logs
    print("Loading no-cycle ICL data...")
    no_cycle_icl_data = generator.load_no_cycle_icl_data_from_logs(
        log_dir, OLMO_JOB_MAP, layer, model_name="OLMo-1B"
    )
    
    if not natural_data and not no_cycle_icl_data:
        print(f"No data found for OLMo")
        return
    
    # Pass OLMo checkpoints to the plotting function
    olmo_checkpoints = list(OLMO_JOB_MAP.keys())
    print(f"\nUsing OLMo checkpoints: {olmo_checkpoints}")
    
    result_path = generator.create_beautiful_dual_alluvial(
        natural_data, no_cycle_icl_data, output_file, checkpoints=olmo_checkpoints
    )
    if result_path:
        print(f"✓ Generated dual plot: {result_path}")

if __name__ == "__main__":
    main()
