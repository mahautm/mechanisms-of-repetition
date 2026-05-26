#!/usr/bin/env python3
import ast
from pathlib import Path
from parrots.cycle_detection import detect_cycles
import pandas as pd

p = Path('outputs/EleutherAI/pythia-1.4b/minipile_alluvial/pile_natural_results.csv')
if not p.exists():
    raise SystemExit('alluvial file missing')

df = pd.read_csv(p)

# parse output_tokens which may be stored as string repr of list
def parse_tokens(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return [int(x) for x in s.strip('[]').split(',') if x.strip()]
        except Exception:
            return []

recomputed = []
for i, row in df.iterrows():
    toks = parse_tokens(row.get('output_tokens',''))
    if not toks:
        recomputed.append((i, False, None))
        continue
    # detect_cycles expects a tensor-like; but supports lists
    c, cs, cc, _ = detect_cycles(toks, return_index=True)
    is_cyc = bool(cc > 0)
    recomputed.append((i, is_cyc, int(cs) if cs is not None else 0))

# compare with original is_cyclical column
orig_flags = df['is_cyclical'].fillna(False).astype(bool)
match = 0
mismatch_idx = []
for (i, rec_flag, rec_len) in recomputed:
    orig = bool(orig_flags.iloc[i])
    if orig == rec_flag:
        match += 1
    else:
        mismatch_idx.append((i, orig, rec_flag, rec_len, df['cycle_length'].iloc[i]))

print('Alluvial rows:', len(df))
print('Recomputed matches:', match)
print('Recomputed mismatches:', len(mismatch_idx))
for m in mismatch_idx[:20]:
    print(m)
