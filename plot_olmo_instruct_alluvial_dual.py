#!/usr/bin/env python3
"""
Generate OLMo Instruction-Tuned Alluvial Plot using existing run_alluvial_dual.py infrastructure
"""

import sys
sys.path.append('/home/mmahaut/projects/parrots')

from pathlib import Path
from run_alluvial_dual import BeautifulDualAlluvial

# OLMo Instruction-Tuned checkpoint to job ID mapping (100 samples)
OLMO_INSTRUCT_JOB_MAP = {
    'step0-tokens0B': '1839787',
    'step288000-tokens603B': '1839788',
    'step577000-tokens1209B': '1839789',
    'step865000-tokens1813B': '1839793',
    'step1165000-tokens2442B': '1839794',
    'step1454000-tokens3048B': '1839795',
}

def main():
    print("OLMo Instruction-Tuned Alluvial Plot Generator")
    print("=" * 50)
    
    log_dir = "logs"
    layer = 12
    output_file = "plots/olmo_instruct_alluvial_layer12.png"
    
    generator = BeautifulDualAlluvial()
    
    print(f"\nProcessing OLMo-1B Instruction-Tuned...")
    
    # Load natural data from logs
    print("Loading natural repetition data...")
    natural_data = generator.load_natural_repetition_data_from_logs(
        log_dir, OLMO_INSTRUCT_JOB_MAP, layer, model_name="OLMo-1B-Instruct"
    )
    
    # Load no-cycle ICL data from logs
    print("Loading no-cycle ICL data...")
    no_cycle_icl_data = generator.load_no_cycle_icl_data_from_logs(
        log_dir, OLMO_INSTRUCT_JOB_MAP, layer, model_name="OLMo-1B-Instruct"
    )
    
    if not natural_data and not no_cycle_icl_data:
        print(f"No data found for OLMo Instruction-Tuned")
        return
    
    # Pass OLMo checkpoints to the plotting function
    olmo_checkpoints = list(OLMO_INSTRUCT_JOB_MAP.keys())
    print(f"\nUsing OLMo Instruction-Tuned checkpoints: {olmo_checkpoints}")
    
    result_path = generator.create_beautiful_dual_alluvial(
        natural_data, no_cycle_icl_data, output_file, checkpoints=olmo_checkpoints, label_prefix="PRE"
    )
    if result_path:
        print(f"✓ Generated dual plot: {result_path}")

if __name__ == "__main__":
    main()
