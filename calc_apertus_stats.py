#!/usr/bin/env python3
import re

jobs = {
    "step50000-tokens210B": "1841716",
    "step500000-tokens2100B": "1841717", 
    "step1000000-tokens4200B": "1841718",
    "step1700000-tokens7232B": "1841719",
    "step2300000-tokens12272B": "1841720",
    "step2627139-tokens15T": "1841721",
}

print("Apertus-8B Layer 24 Repetition Statistics")
print("=" * 80)

for checkpoint, job_id in jobs.items():
    log_file = f"/home/mmahaut/projects/parrots/logs/apertus_attention_{job_id}.out"
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract data
    data_match = re.search(r'layer 24 data index: \[(.*?)\]', content)
    rep_match = re.search(r'layer 24 repetition index: \[(.*?)\]', content)
    
    if data_match and rep_match:
        data_str = data_match.group(1)
        rep_str = rep_match.group(1)
        
        data_indices = [int(x.strip()) for x in data_str.split(',') if x.strip()]
        rep_indices = [int(x.strip()) for x in rep_str.split(',') if x.strip()]
        
        # Count how many repetitions are > 0 (non-repeating is represented by larger values like 128)
        repeating_count = sum(1 for r in rep_indices if r < 128)
        total_count = len(rep_indices)
        percentage = (repeating_count / total_count * 100) if total_count > 0 else 0
        
        # Also calculate different thresholds
        very_repeating = sum(1 for r in rep_indices if r < 50)
        moderately_repeating = sum(1 for r in rep_indices if 50 <= r < 100)
        slightly_repeating = sum(1 for r in rep_indices if 100 <= r < 128)
        non_repeating = sum(1 for r in rep_indices if r >= 128)
        
        print(f"\n{checkpoint}:")
        print(f"  Total samples: {total_count}")
        print(f"  Very repeating (< 50 tokens): {very_repeating} ({very_repeating/total_count*100:.1f}%)")
        print(f"  Moderately (50-99): {moderately_repeating} ({moderately_repeating/total_count*100:.1f}%)")
        print(f"  Slightly (100-127): {slightly_repeating} ({slightly_repeating/total_count*100:.1f}%)")
        print(f"  Non-repeating (>= 128): {non_repeating} ({non_repeating/total_count*100:.1f}%)")
        print(f"  Overall repeating (< 128): {percentage:.1f}%")
