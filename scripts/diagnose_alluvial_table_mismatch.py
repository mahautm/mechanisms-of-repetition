#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# Paths
alluvial = Path('outputs/EleutherAI/pythia-1.4b/minipile_alluvial/pile_natural_results.csv')
table = Path('plots/cycle_detection_EleutherAI_pythia-1_4b.csv')

if not alluvial.exists():
    raise SystemExit('Alluvial CSV not found')
if not table.exists():
    raise SystemExit('Table CSV not found')

adf = pd.read_csv(alluvial)
tdf = pd.read_csv(table)

# Normalize column names
# alluvial: condition,input,prompt_length,output_tokens,cycle_length,cycle,is_cyclical
# table: model,generated,cycle,cycle_length,cycle_count,elapsed_seconds

print('Alluvial rows:', len(adf))
print('Table rows:', len(tdf))

# Build simple keys by prompt prefix
adf['prompt_key'] = adf['input'].astype(str).str[:200]
tdf['prompt_key'] = tdf['generated'].astype(str).str[:200]  # generated includes prompt+output; we try to match by prefix

# Try to match by prompt prefix appearing in generated
matches = 0
mismatch_examples = []
for i, arow in adf.iterrows():
    key = arow['prompt_key']
    # find rows in table where generated startswith key
    cand = tdf[tdf['generated'].str.startswith(key, na=False)]
    if cand.empty:
        continue
    matches += 1
    # take first candidate
    grow = cand.iloc[0]
    allu_flag = bool(arow.get('is_cyclical') == True or int(arow.get('is_cyclical', 0)) == 1)
    table_flag = int(grow.get('cycle_count', 0)) > 0
    if allu_flag != table_flag and len(mismatch_examples) < 10:
        mismatch_examples.append({
            'prompt_prefix': key[:200],
            'alluvial_is_cyclical': allu_flag,
            'table_cycle_count': int(grow.get('cycle_count', 0)),
            'alluvial_cycle_length': arow.get('cycle_length'),
            'table_cycle_length': grow.get('cycle_length')
        })

print('Matched prompts between datasets:', matches)
print('Collected mismatches:', len(mismatch_examples))
for ex in mismatch_examples:
    print('\n--- MISMATCH ---')
    print('Prompt prefix:', ex['prompt_prefix'])
    print('Alluvial cyclical:', ex['alluvial_is_cyclical'], 'cycle_length', ex['alluvial_cycle_length'])
    print('Table cyclical:', ex['table_cycle_count'], 'cycle_length', ex['table_cycle_length'])

# Summary of flags on intersection
# Build dataframe of matched pairs
pairs = []
for i, arow in adf.iterrows():
    key = arow['prompt_key']
    cand = tdf[tdf['generated'].str.startswith(key, na=False)]
    if cand.empty:
        continue
    grow = cand.iloc[0]
    allu_flag = bool(arow.get('is_cyclical') == True or int(arow.get('is_cyclical', 0)) == 1)
    table_flag = int(grow.get('cycle_count', 0)) > 0
    pairs.append((allu_flag, table_flag))

if pairs:
    import collections
    c = collections.Counter(pairs)
    print('\nContingency (alluvial_flag, table_flag):')
    for k, v in c.items():
        print(k, v)
else:
    print('No matching pairs found to compare')
