#!/usr/bin/env python3
"""Teacher-forced head ablation on paired Natural/ICL prompts.

Replaces the binary "does the generation repeat" outcome with continuous, per-position
measures on a fixed token sequence, so single-head effects are measurable well below the
ceiling of the generation outcome.

Stages (one checkpoint per call):
  baseline  Build the paired prompts, greedily generate a fixed continuation, detect its cycle,
            and save tokens, regions, per-head attention scores (induction, duplicate-token,
            previous-token, first-token sink) and per-head mean activations (for mean-ablation).
  single    For every head of the requested layers, ablate it and measure, on the cached tokens,
            the change in log-probability of the actual next token, in logit margin over the best
            other token, and whether the top prediction flips. Results are averaged per sequence
            and region.
  group     Same measures, ablating sets of heads at once: the top-k heads by a score
            (induction, previous-token) and size-matched random sets.

Regions (positions are the *target* token index in the full sequence):
  prompt2   tokens 33..63 of the prompt. For ICL this is the copied half (induction territory);
            for Natural it is ordinary running text from the same source.
  pre       generated tokens before the trailing cycle starts.
  cycle     generated tokens from the second full cycle onward (copyable from the previous cycle).
  cont      all generated tokens.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ablation_head_cycle_icl_natural import (
    build_paired_condition_prompts,
    get_attention_module,
    get_output_projection,
    get_transformer_layers,
    infer_num_heads,
    load_model_and_tokenizer,
    load_raw_minipile_texts,
    set_seed,
)
from parrots.cycle_detection import detect_cycles

REGIONS = ["prompt2", "pre", "cycle", "cont"]


# ---------------------------------------------------------------------------- prompts / tokens

def build_sequences(args, tokenizer):
    """Paired prompts (tokenizer-only, identical at every checkpoint), Natural rows then ICL rows."""
    raw = load_raw_minipile_texts(args.raw_pool_size, args.seed, args.natural_dataset)
    src, nat, icl = build_paired_condition_prompts(tokenizer, raw, args.raw_prompt_tokens)
    src, nat, icl = src[: args.n_samples], nat[: args.n_samples], icl[: args.n_samples]
    prompts = torch.tensor(nat + icl, dtype=torch.long)
    cond = np.array(["natural"] * len(nat) + ["icl"] * len(icl))
    source = np.array(src + src)
    return prompts, cond, source


@torch.no_grad()
def generate_continuations(model, prompts, max_new_tokens, batch_size, device):
    """Greedy, fixed-length continuations. EOS is suppressed (min_new_tokens) so every row has
    the same length and teacher forcing needs no padding."""
    out = []
    for i in range(0, len(prompts), batch_size):
        x = prompts[i : i + batch_size].to(device)
        g = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            pad_token_id=0,
        )
        out.append(g.cpu())
    return torch.cat(out)


def region_masks(tokens, prompt_len, k):
    """Boolean masks over target positions j (predicting tokens[:, j+1]), one per region."""
    n, t = tokens.shape
    tgt = np.arange(1, t)
    masks = {r: np.zeros((n, t - 1), dtype=bool) for r in REGIONS}
    starts, sizes, counts = [], [], []
    masks["prompt2"][:, (tgt >= k + 1) & (tgt < prompt_len)] = True
    masks["cont"][:, tgt >= prompt_len] = True
    for i in range(n):
        cont = tokens[i, prompt_len:].tolist()
        _, size, count, start = detect_cycles(cont, return_index=True)
        repeats = count > 1
        starts.append(start if repeats else -1)
        sizes.append(size if repeats else 0)
        counts.append(count if repeats else 0)
        if repeats:
            masks["pre"][i, (tgt >= prompt_len) & (tgt < prompt_len + start)] = True
            masks["cycle"][i, tgt >= prompt_len + start + size] = True
        else:
            masks["pre"][i, tgt >= prompt_len] = True
    return masks, np.array(starts), np.array(sizes), np.array(counts)


# ---------------------------------------------------------------------------- model plumbing

def head_slices(heads: List[int], head_dim: int) -> torch.Tensor:
    return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads])


def install_ablation(model, plan: Dict[int, List[int]], mode: str, mean_z: Optional[torch.Tensor]):
    """Ablate heads by editing the *input* of each layer's output projection (the per-head
    concatenation, before heads are mixed). plan maps layer -> list of heads."""
    layers = get_transformer_layers(model)
    handles = []
    for layer_idx, heads in plan.items():
        attn = get_attention_module(layers[layer_idx])
        proj = get_output_projection(attn)
        n_heads = infer_num_heads(model, attn)
        d = proj.in_features if hasattr(proj, "in_features") else model.config.hidden_size
        idx = head_slices(heads, d // n_heads)
        fill = mean_z[layer_idx, idx] if mode == "mean" else None

        def hook(module, args, idx=idx, fill=fill):
            x = args[0].clone()
            ii = idx.to(x.device)
            x[..., ii] = 0.0 if fill is None else fill.to(device=x.device, dtype=x.dtype)
            return (x, *args[1:])

        handles.append(proj.register_forward_pre_hook(hook))
    return handles


@torch.no_grad()
def token_metrics(model, tokens, batch_size, device):
    """Per target position: log p(actual next token), logit margin over the best other token,
    and whether the argmax differs from the actual next token."""
    lp_all, margin_all, flip_all = [], [], []
    for i in range(0, len(tokens), batch_size):
        x = tokens[i : i + batch_size].to(device)
        logits = model(input_ids=x).logits[:, :-1].float()
        tgt = x[:, 1:]
        lp = torch.log_softmax(logits, -1).gather(-1, tgt[..., None])[..., 0]
        top2 = logits.topk(2, dim=-1)
        tgt_logit = logits.gather(-1, tgt[..., None])[..., 0]
        is_top = top2.indices[..., 0] == tgt
        best_other = torch.where(is_top, top2.values[..., 1], top2.values[..., 0])
        lp_all.append(lp.cpu())
        margin_all.append((tgt_logit - best_other).cpu())
        flip_all.append((~is_top).cpu())
        del logits
    return torch.cat(lp_all).numpy(), torch.cat(margin_all).numpy(), torch.cat(flip_all).numpy()


def region_means(values, masks):
    out = {}
    for r, m in masks.items():
        n = m.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = np.where(n > 0, (values * m).sum(1) / n, np.nan)
    return out


@torch.no_grad()
def attention_scores_and_means(model, prompts, cond, k, batch_size, device):
    """Per-head attention scores on the prompts and per-layer mean of the output-projection input.

    induction      ICL second copy: attention from position k+i to i+1 (token after the earlier copy)
    duplicate      ICL second copy: attention from position k+i to i (earlier copy of the same token)
    prev_token     Natural prompts: attention to the immediately preceding position
    first_token    Natural prompts: attention to position 0 (sink)
    """
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    n_heads = infer_num_heads(model, get_attention_module(layers[0]))
    sums = {s: torch.zeros(n_layers, n_heads, dtype=torch.float64) for s in ["induction", "duplicate", "prev_token", "first_token"]}
    counts = {"icl": 0, "natural": 0}
    z_sum, z_n = None, 0
    captured = {}

    def make_cap(li):
        def cap(module, args):
            captured[li] = args[0].detach().float().sum(dim=(0, 1)).double().cpu()
        return cap

    handles = [get_output_projection(get_attention_module(l)).register_forward_pre_hook(make_cap(li)) for li, l in enumerate(layers)]
    q_ind = torch.arange(k, 2 * k - 1)
    q_all = torch.arange(1, 2 * k)
    try:
        for c in ["icl", "natural"]:
            rows = torch.from_numpy(np.where(cond == c)[0])
            for i in range(0, len(rows), batch_size):
                x = prompts[rows[i : i + batch_size]].to(device)
                att = model(input_ids=x, output_attentions=True).attentions  # per layer: B, H, T, T
                if z_sum is None:
                    z_sum = torch.zeros(n_layers, captured[0].numel(), dtype=torch.float64)
                for li in range(n_layers):
                    z_sum[li] += captured[li]
                    a = att[li].float()
                    if c == "icl":
                        sums["induction"][li] += a[:, :, q_ind, q_ind - k + 1].mean(-1).sum(0).double().cpu()
                        sums["duplicate"][li] += a[:, :, q_ind, q_ind - k].mean(-1).sum(0).double().cpu()
                    else:
                        sums["prev_token"][li] += a[:, :, q_all, q_all - 1].mean(-1).sum(0).double().cpu()
                        sums["first_token"][li] += a[:, :, q_all, 0].mean(-1).sum(0).double().cpu()
                z_n += x.shape[0] * x.shape[1]
                counts[c] += x.shape[0]
    finally:
        for h in handles:
            h.remove()
    rows = []
    for li in range(n_layers):
        for h in range(n_heads):
            rows.append({
                "layer": li, "head": h,
                "induction": float(sums["induction"][li, h] / counts["icl"]),
                "duplicate": float(sums["duplicate"][li, h] / counts["icl"]),
                "prev_token": float(sums["prev_token"][li, h] / counts["natural"]),
                "first_token": float(sums["first_token"][li, h] / counts["natural"]),
            })
    return pd.DataFrame(rows), (z_sum / z_n).float()


def set_eager(model):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"


def set_default_attn(model):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("sdpa")
    else:
        model.config._attn_implementation = "sdpa"


def compute_scores(args, model, prompts, cond, device, ckpt_dir):
    """Attention scores and mean activations in fp32: fp16 eager attention overflows to NaN for
    some heads of trained Pythia checkpoints."""
    model.float()
    set_eager(model)
    try:
        scores, mean_z = attention_scores_and_means(model, prompts, cond, args.raw_prompt_tokens, args.batch_size, device)
    finally:
        set_default_attn(model)
        model.half()
    n_nan = int(scores[["induction", "prev_token"]].isna().any(axis=1).sum())
    if n_nan or not torch.isfinite(mean_z).all():
        raise RuntimeError(f"non-finite attention scores for {n_nan} heads")
    scores.insert(0, "checkpoint", args.checkpoint)
    scores.to_csv(ckpt_dir / "head_scores.csv", index=False)
    return scores, mean_z


def run_scores(args, model, device, ckpt_dir):
    """Recompute head scores and mean activations for an existing baseline (tokens unchanged)."""
    cache = torch.load(ckpt_dir / "baseline.pt", weights_only=False)
    prompts = cache["tokens"][:, : cache["prompt_len"]]
    _, cache["mean_z"] = compute_scores(args, model, prompts, cache["cond"], device, ckpt_dir)
    torch.save(cache, ckpt_dir / "baseline.pt")
    print(f"  rewrote head_scores.csv and mean_z in {ckpt_dir}")


# ---------------------------------------------------------------------------- stages

def run_baseline(args, model, tokenizer, device, ckpt_dir):
    prompts, cond, source = build_sequences(args, tokenizer)
    print(f"  paired prompts: {len(prompts)} rows ({(cond == 'natural').sum()} per condition)")
    tokens = generate_continuations(model, prompts, args.max_new_tokens, args.gen_batch_size, device)
    masks, starts, sizes, counts = region_masks(tokens.numpy(), prompts.shape[1], args.raw_prompt_tokens)

    scores, mean_z = compute_scores(args, model, prompts, cond, device, ckpt_dir)

    torch.save({
        "tokens": tokens, "cond": cond, "source": source, "prompt_len": prompts.shape[1],
        "masks": masks, "cycle_start": starts, "cycle_size": sizes, "cycle_count": counts,
        "mean_z": mean_z, "args": vars(args),
    }, ckpt_dir / "baseline.pt")

    lp, margin, flip = token_metrics(model, tokens, args.batch_size, device)
    desc = pd.DataFrame({"cond": cond, "source": source, "repeats": counts > 1, "cycle_start": starts, "cycle_size": sizes})
    for name, v in [("logprob", lp), ("margin", margin), ("flip", flip.astype(float))]:
        for r, m in region_means(v, masks).items():
            desc[f"base_{name}_{r}"] = m
    desc.insert(0, "checkpoint", args.checkpoint)
    desc.to_csv(ckpt_dir / "baseline_per_sequence.csv.gz", index=False)
    print(desc.groupby("cond")[["repeats", "cycle_size", "base_logprob_cycle", "base_logprob_prompt2", "base_flip_pre"]].mean())


def ablation_rows(args, model, cache, device, plans, base):
    masks = cache["masks"]
    rows = []
    for label, plan in plans:
        handles = install_ablation(model, plan, args.ablation, cache["mean_z"])
        try:
            lp, margin, flip = token_metrics(model, cache["tokens"], args.batch_size, device)
        finally:
            for h in handles:
                h.remove()
        res = {"logprob": region_means(lp - base["logprob"], masks),
               "margin": region_means(margin - base["margin"], masks),
               "flip": region_means(flip.astype(float) - base["flip"], masks)}
        for r in REGIONS:
            keep = ~np.isnan(res["logprob"][r])
            df = pd.DataFrame({
                "cond": cache["cond"][keep], "source": cache["source"][keep], "region": r,
                "d_logprob": res["logprob"][r][keep], "d_margin": res["margin"][r][keep], "d_flip": res["flip"][r][keep],
            })
            for key, val in label.items():
                df.insert(0, key, val)
            rows.append(df)
        print(f"    {label}: d_logprob cycle ICL={np.nanmean(res['logprob']['cycle'][cache['cond'] == 'icl']):+.4f} "
              f"Nat={np.nanmean(res['logprob']['cycle'][cache['cond'] == 'natural']):+.4f}", flush=True)
    return pd.concat(rows, ignore_index=True)


def base_metrics(args, model, cache, device):
    lp, margin, flip = token_metrics(model, cache["tokens"], args.batch_size, device)
    return {"logprob": lp, "margin": margin, "flip": flip.astype(float)}


def run_single(args, model, device, ckpt_dir):
    cache = torch.load(ckpt_dir / "baseline.pt", weights_only=False)
    layers = get_transformer_layers(model)
    n_heads = infer_num_heads(model, get_attention_module(layers[0]))
    layer_ids = args.layers or list(range(len(layers)))
    base = base_metrics(args, model, cache, device)
    for li in layer_ids:
        out = ckpt_dir / f"single_{args.ablation}_L{li}.csv.gz"
        if out.exists() and not args.overwrite:
            print(f"  skip existing {out}")
            continue
        plans = [({"checkpoint": args.checkpoint, "ablation": args.ablation, "layer": li, "head": h}, {li: [h]}) for h in range(n_heads)]
        ablation_rows(args, model, cache, device, plans, base).to_csv(out, index=False)
        print(f"  saved {out}")


def run_group(args, model, device, ckpt_dir):
    cache = torch.load(ckpt_dir / "baseline.pt", weights_only=False)
    scores = pd.read_csv(ckpt_dir / "head_scores.csv")
    base = base_metrics(args, model, cache, device)
    rng = np.random.default_rng(args.seed)
    all_heads = list(zip(scores.layer, scores["head"]))
    plans = []
    for k in args.group_ks:
        for score in args.group_scores:
            top = scores.sort_values(score, ascending=False).head(k)
            plans.append((f"top_{score}", 0, list(zip(top.layer, top["head"]))))
        for r in range(args.n_random):
            pick = rng.choice(len(all_heads), size=k, replace=False)
            plans.append(("random", r, [all_heads[i] for i in pick]))
    built = []
    for kind, rep, heads in plans:
        plan: Dict[int, List[int]] = {}
        for layer, head in heads:
            plan.setdefault(int(layer), []).append(int(head))
        label = {"checkpoint": args.checkpoint, "ablation": args.ablation, "set": kind, "rep": rep, "k": len(heads),
                 "heads": json.dumps(sorted([f"L{l}H{h}" for l, h in heads]))}
        built.append((label, plan))
    out = ckpt_dir / f"group_{args.ablation}.csv.gz"
    ablation_rows(args, model, cache, device, built, base).to_csv(out, index=False)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["baseline", "scores", "single", "group"], required=True)
    ap.add_argument("--model_name", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", default="outputs_teacher_forced/pythia14b")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="single stage: layers to sweep (default all)")
    ap.add_argument("--ablation", choices=["zero", "mean"], default="zero")
    ap.add_argument("--group_ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--group_scores", nargs="+", default=["induction", "prev_token"])
    ap.add_argument("--n_random", type=int, default=5)
    # Same prompt construction as the paired generation sweep (outputs_ablation_paired_fp16).
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--raw_pool_size", type=int, default=1500)
    ap.add_argument("--raw_prompt_tokens", type=int, default=32)
    ap.add_argument("--natural_dataset", default="JeanKaddour/minipile")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gen_batch_size", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    ckpt_dir = Path(args.output_dir) / args.checkpoint.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.checkpoint, use_bnb=False)
    model.to(device).eval()
    print(f"[{args.stage}] {args.model_name}@{args.checkpoint} ablation={args.ablation} -> {ckpt_dir}")

    if args.stage == "baseline":
        run_baseline(args, model, tokenizer, device, ckpt_dir)
    elif args.stage == "scores":
        run_scores(args, model, device, ckpt_dir)
    elif args.stage == "single":
        run_single(args, model, device, ckpt_dir)
    else:
        run_group(args, model, device, ckpt_dir)


if __name__ == "__main__":
    main()
