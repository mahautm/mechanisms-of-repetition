import pandas as pd

df = pd.read_csv('plots/cycle_intro_stats_by_model.csv')

# Drop the quick test
df = df[~df['model_key'].str.contains('quick_test')]

# Map keys to nice names
def nice_name(key):
    if 'Apertus-8B' in key: return 'Apertus 8B'
    if 'opt-1.3b' in key: return 'OPT 1.3B'
    if 'Llama-3.2-1B' in key: return 'Llama-3.2 1B'
    if 'OLMo-1B' in key: return 'OLMo 1B'
    if 'pythia-1.4b' in key and 'human_lama' in key: return 'Pythia 1.4B'
    if 'pythia-6.9b' in key and 'human_lama' in key: return 'Pythia 6.9B'
    return None

df['Model'] = df['model_key'].apply(nice_name)

# Filter out rows where Model is None (i.e. Pythia non-human_lama datasets to keep everything 1:1 comparable)
df = df.dropna(subset=['Model']).copy()

# Sort
df = df.sort_values(by='pct_cyclic', ascending=False)

print("\\begin{table}[htpb]")
print("\\centering")
print("\\begin{tabular}{lrrrrrr}")
print("\\toprule")
print("\\textbf{Model} & \\textbf{Total Seq.} & \\textbf{Cyclic Seq.} & \\textbf{\\% Cyclic} & \\textbf{Mean Size} & \\textbf{Mean Count} & \\textbf{First Cycle (toks)} \\\\")
print("\\midrule")

for _, row in df.iterrows():
    m = row['Model']
    t = int(row['n_total'])
    c = int(row['n_cyclic'])
    p = row['pct_cyclic']
    ms = row['mean_cycle_size']
    mc = row['mean_cycle_count']
    fc = row['mean_tokens_to_first_cycle']
    print(f"{m:<13} & {t:,} & {c:,} & {p:.2f}\\% & {ms:.2f} & {mc:.2f} & {fc:.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Descriptive statistics of repetition cycles per model on the Human LAMA dataset with 512 max new tokens.}")
print("\\label{tab:cycle_stats_model}")
print("\\end{table}")
