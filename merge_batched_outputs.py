#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path('/home/mmahaut/projects/parrots/outputs_multihead_full_new')
MODEL = 'EleutherAI/pythia-1.4b'
LAYER = 23
CHECKPOINTS = [ 'step1','step1000','step5000','step10000','step50000','step100000','step140000','step143000']

pattern = re.compile(r'off(\d+)')

for cp in CHECKPOINTS:
    cp_path = ROOT / MODEL / cp / f'layer_{LAYER}'
    if not cp_path.exists():
        print(f'missing path {cp_path}')
        continue
    # find batched files for cyc2
    files = list(cp_path.glob('full_analysis_cyc2*_off*.out'))
    if not files:
        # fall back to existing full file
        full = cp_path / 'full_analysis_cyc2_full.out'
        if full.exists():
            print(f'Using existing full file for {cp}')
            continue
        else:
            print(f'No batched or full files for {cp}, skipping')
            continue
    data_indices = []
    repetition_indices = set()
    no_cycle_icl_indices = set()
    for f in files:
        m = pattern.search(f.name)
        if not m:
            continue
        offset = int(m.group(1))
        # read contents
        txt = f.read_text()
        # helper to parse list
        def parse_list(prefix):
            m = re.search(rf"layer {LAYER} {prefix}: \[(.*?)\]", txt, re.S)
            if not m:
                return []
            s = m.group(1).strip()
            if not s:
                return []
            return [int(x.strip()) for x in s.split(',') if x.strip()]
        di = parse_list('data index')
        ri = parse_list('repetition index')
        ni = parse_list('no-cycle icl index')
        # offset data indices and repetition and no-cycle
        data_indices.extend([offset + x for x in di])
        repetition_indices.update([offset + x for x in ri])
        no_cycle_icl_indices.update([offset + x for x in ni])
    data_indices = sorted(set(data_indices))
    repetition_indices = sorted(repetition_indices)
    no_cycle_icl_indices = sorted(no_cycle_icl_indices)
    # write combined full file
    out_file = cp_path / 'full_analysis_cyc2_full.out'
    with open(out_file, 'w') as out:
        out.write(f"layer {LAYER} cycle count: {len(repetition_indices)}\n")
        out.write(f"layer {LAYER} data index: {data_indices}\n")
        out.write(f"layer {LAYER} repetition index: {repetition_indices}\n")
        out.write(f"layer {LAYER} no-cycle icl index: {no_cycle_icl_indices}\n")
        out.write(f"layer {LAYER} no-cycle icl cycle count: {len(no_cycle_icl_indices)}\n")
    print(f'Wrote combined file for {cp}: {out_file} (data {len(data_indices)}, reps {len(repetition_indices)})')
