import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import glob 

def detect_cycles(tokens, return_index=False, pad_token_id=None):
    # Only detect cycles that repeat until the end of the sequence.
    # Cycles do not have to fit perfectly; partial cycles at the start are allowed.
    cycle_min_length = 2
    cycle_max_length = 500
    cycle = None
    cycle_size = 0
    cycle_count = 0
    cycle_start_index = -1

    # Convert to list if it's a tensor
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()

    n = len(tokens)
    # Start from the smallest allowed cycle size and increase
    for size in range(cycle_min_length, min(cycle_max_length, n // 2) + 1):
        # Take the last 'size' tokens as the candidate cycle
        candidate = tokens[-size:]  # Already a list, no need for .tolist()
        if pad_token_id is not None:
            # check if the candidate contains only pad tokens
            if all(token == pad_token_id for token in candidate):
                continue
        # Initialize count of consecutive cycles found at the end
        count = 0
        # Start checking from the end, moving backwards in steps of 'size'
        i = n - size
        while i >= 0 and tokens[i:i+size] == candidate:  # Direct list comparison
            count += 1
            i -= size
        # If the candidate repeats more than once until the end, we found a cycle
        # NOTE: Do NOT treat a single whole-sequence match as a cycle — require
        # at least two repetitions to avoid false positives for outputs that
        # are only a single instance of the candidate.
        if count > 1:
            cycle = candidate
            cycle_size = size
            cycle_count = count
            # The index where the first full cycle starts (may leave a partial at the start)
            cycle_start_index = i + size
            break

    if return_index:
        return cycle, cycle_size, cycle_count, cycle_start_index
    return cycle, cycle_size, cycle_count


def process_generation_column(df, generation_col='generated', tokenizer=None):
    tqdm.pandas()
    # deal with nan values
    df[generation_col] = df[generation_col].fillna("")
    df['tokens'] = df.progress_apply(lambda x: tokenizer(x[generation_col], truncation=True, padding=True)['input_ids'], axis=1)
    print(df['tokens'])
    df['cycle'], df['cycle_size'], df['cycle_count'] = zip(*df['tokens'].progress_apply(detect_cycles))
    df['cycle'] = df['cycle'].progress_apply(tokenizer.decode)
    # drop tokens column
    df = df.drop(columns=['tokens'])
    return df

if __name__ == '__main__':
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")    
    for file in glob.glob("/home/mmahaut/projects/parrots/outputs/*/*/slot_filling_results.csv"):
        df = pd.read_csv(file)
        df = df.dropna(subset=['generated'])
        df = process_generation_column(df, tokenizer=tokenizer)
        df.to_csv(file.replace('slot_filling_results.csv', 'slot_filling_results_with_cycles.csv'), index=False)