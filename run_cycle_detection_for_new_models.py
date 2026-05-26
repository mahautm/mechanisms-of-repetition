#!/usr/bin/env python3
"""Post-process new model slot-filling outputs with cycle detection."""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from parrots.cycle_detection import detect_cycles


def process_one(model_name: str, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"Skip {model_name}: missing {csv_path}")
        return

    out_path = csv_path.with_name("slot_filling_results_with_cycles.csv")
    if out_path.exists():
        print(f"Exists {out_path}, skipping")
        return

    print(f"Processing {model_name}: {csv_path}")
    df = pd.read_csv(csv_path)
    if "generated" not in df.columns:
        raise ValueError(f"No 'generated' column in {csv_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    df = df.copy()

    def _tok(text: str):
        return tokenizer(text, truncation=True, padding=True)["input_ids"]

    df["generated"] = df["generated"].fillna("")
    df["tokens"] = df["generated"].astype(str).apply(_tok)
    cycles = df["tokens"].apply(detect_cycles)
    df["cycle_raw"] = cycles.apply(lambda x: x[0])
    df["cycle_size"] = cycles.apply(lambda x: int(x[1]) if x[1] is not None else 0)
    df["cycle_count"] = cycles.apply(lambda x: int(x[2]) if x[2] is not None else 0)
    df["cycle"] = df["cycle_raw"].apply(lambda c: tokenizer.decode(c) if c else "")
    df = df.drop(columns=["tokens", "cycle_raw"])
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main() -> None:
    targets = {
        "meta-llama/Llama-3.2-1B": Path(
            "/home/mmahaut/projects/parrots/outputs/meta-llama/Llama-3.2-1B_512_sf/slot_filling_results.csv"
        ),
        "facebook/opt-1.3b": Path(
            "/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_512_sf/slot_filling_results.csv"
        ),
        "allenai/OLMo-1B-hf": Path(
            "/home/mmahaut/projects/parrots/outputs/allenai/OLMo-1B-hf_512_sf/slot_filling_results.csv"
        ),
        "swiss-ai/Apertus-8B-2509": Path(
            "/home/mmahaut/projects/parrots/outputs/swiss-ai/Apertus-8B-2509_512_sf/slot_filling_results.csv"
        ),
    }

    for model_name, csv_path in targets.items():
        process_one(model_name, csv_path)


if __name__ == "__main__":
    main()
