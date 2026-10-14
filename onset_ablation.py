#!/usr/bin/env python3
"""Cycle entry under group ablation: does removing heads change *whether and when* a cycle starts?

Teacher forcing (teacher_forced_ablation.py) only measures how well the model sustains the
continuation it actually produced, so it cannot see a cycle that the ablated model would enter
but the intact model does not (e.g. the suppressor L12H0). Here we regenerate the continuation
of every paired prompt with heads ablated and record, per sequence:
  repeats     the generation ends in a trailing cycle
  onset       generated tokens before the first full cycle (censored at max_new_tokens if none)
  cycle_size  cycle length in tokens

Head sets (per checkpoint, from head_scores.csv of the teacher-forced baseline):
  none, L12H0, top-k by induction score, top-k by previous-token score, size-matched random sets.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ablation_head_cycle_icl_natural import load_model_and_tokenizer, set_seed
from parrots.cycle_detection import detect_cycles
from teacher_forced_ablation import build_sequences, generate_continuations, install_ablation


def cycle_stats(tokens: torch.Tensor, prompt_len: int, max_new: int):
    rows = []
    for seq in tokens[:, prompt_len:].tolist():
        _, size, count, start = detect_cycles(seq, return_index=True)
        repeats = count > 1
        rows.append((repeats, start if repeats else max_new, size if repeats else 0))
    return rows


def head_sets(scores: pd.DataFrame, ks: List[int], random_ks: List[int], n_random: int, seed: int):
    sets = [("none", 0, []), ("L12H0", 0, [(12, 0)])]
    for score in ["induction", "prev_token"]:
        ranked = scores.sort_values(score, ascending=False)
        for k in ks:
            top = ranked.head(k)
            sets.append((f"top_{score}", 0, list(zip(top.layer.astype(int), top["head"].astype(int)))))
    rng = np.random.default_rng(seed)
    all_heads = list(zip(scores.layer.astype(int), scores["head"].astype(int)))
    for k in random_ks:
        for r in range(n_random):
            sets.append(("random", r, [all_heads[i] for i in rng.choice(len(all_heads), size=k, replace=False)]))
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scores_dir", default="outputs_teacher_forced/pythia14b")
    ap.add_argument("--output_dir", default="outputs_onset/pythia14b")
    ap.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--random_ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--set_index", type=int, nargs="*", default=None, help="Run only these entries of the set list")
    # Same prompts and continuation length as the teacher-forced run.
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--raw_pool_size", type=int, default=1500)
    ap.add_argument("--raw_prompt_tokens", type=int, default=32)
    ap.add_argument("--natural_dataset", default="JeanKaddour/minipile")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--gen_batch_size", type=int, default=100)
    args = ap.parse_args()

    set_seed(args.seed)
    ck = args.checkpoint.replace("/", "_")
    out_dir = Path(args.output_dir) / ck
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_ck = Path(args.scores_dir) / ck
    scores = pd.read_csv(scores_ck / "head_scores.csv")
    mean_z = torch.load(scores_ck / "baseline.pt", weights_only=False)["mean_z"] if args.ablation == "mean" else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.checkpoint, use_bnb=False)
    model.to(device).eval()
    prompts, cond, source = build_sequences(args, tokenizer)

    sets = head_sets(scores, args.ks, args.random_ks, args.n_random, args.seed)
    chosen = args.set_index if args.set_index else range(len(sets))
    for i in chosen:
        kind, rep, heads = sets[i]
        out = out_dir / f"onset_{args.ablation}_{i:03d}.csv.gz"
        if out.exists():
            print(f"  skip existing {out}")
            continue
        plan: Dict[int, List[int]] = {}
        for layer, head in heads:
            plan.setdefault(layer, []).append(head)
        handles = install_ablation(model, plan, args.ablation, mean_z) if plan else []
        try:
            tokens = generate_continuations(model, prompts, args.max_new_tokens, args.gen_batch_size, device)
        finally:
            for h in handles:
                h.remove()
        stats = cycle_stats(tokens, prompts.shape[1], args.max_new_tokens)
        df = pd.DataFrame(stats, columns=["repeats", "onset", "cycle_size"])
        df.insert(0, "source", source)
        df.insert(0, "cond", cond)
        for key, val in [("heads", json.dumps(sorted(f"L{l}H{h}" for l, h in heads))), ("k", len(heads)), ("rep", rep),
                         ("set", kind), ("ablation", args.ablation), ("checkpoint", args.checkpoint)]:
            df.insert(0, key, val)
        df.to_csv(out, index=False)
        g = df.groupby("cond").agg(repeats=("repeats", "mean"), onset=("onset", "median"))
        print(f"  [{i}] {kind} k={len(heads)} rep={rep}: " + "  ".join(
            f"{c}: rep={r.repeats:.3f} onset_med={r.onset:.0f}" for c, r in g.iterrows()), flush=True)


if __name__ == "__main__":
    main()
