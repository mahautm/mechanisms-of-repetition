#!/usr/bin/env python3
"""Generate condition-aware conclusions for ablation results."""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


CHECKPOINT_ORDER = ["step1", "step1000", "step5000", "step10000", "step100000", "steplatest"]


def top_heads_block(rankings: pd.DataFrame, condition: str, positive: bool, top_k: int = 5) -> List[str]:
	subset = rankings[rankings["condition"] == condition].copy()
	title = f"{condition.capitalize()} Top {'Suppressor' if positive else 'Promoter'} Heads"
	if positive:
		subset = subset[subset["mean_rep_rate_delta"] > 0]
	else:
		subset = subset[subset["mean_rep_rate_delta"] < 0]

	lines = [f"## {title}"]
	if subset.empty:
		fallback = rankings[rankings["condition"] == condition].sort_values("causal_score", ascending=False).head(top_k)
		if fallback.empty:
			lines.append(f"- none ({condition} rows missing)")
		else:
			lines.append(f"- no non-zero Delta repetition rate rows for {condition}; listing top cycle-shift heads:")
			for _, row in fallback.iterrows():
				lines.append(
					f"  - L{int(row['layer'])}H{int(row['head'])}: "
					f"Delta rep={row['mean_rep_rate_delta']:.4f}, "
					f"Delta cycle={row['mean_cycle_count_delta']:.2f}, score={row['causal_score']:.3f}"
				)
		lines.append("")
		return lines

	for _, row in subset.head(top_k).iterrows():
		lines.append(
			f"- L{int(row['layer'])}H{int(row['head'])}: "
			f"Delta rep={row['mean_rep_rate_delta']:.4f} +- {row['std_rep_rate_delta']:.4f}, "
			f"Delta cycle={row['mean_cycle_count_delta']:.2f}, score={row['causal_score']:.3f}"
		)
	lines.append("")
	return lines


def checkpoint_block(combined: pd.DataFrame, condition: str) -> List[str]:
	subset = combined[combined["condition"] == condition].copy()
	lines = [f"## {condition.capitalize()} Checkpoint Evolution"]
	if subset.empty:
		lines.append("- no data")
		lines.append("")
		return lines

	present = subset["checkpoint"].dropna().astype(str).unique().tolist()
	extra = [c for c in present if c not in CHECKPOINT_ORDER]
	ordered = CHECKPOINT_ORDER + sorted(extra)
	subset["checkpoint"] = pd.Categorical(subset["checkpoint"], categories=ordered, ordered=True)
	summary = (
		subset.groupby("checkpoint", as_index=False, observed=True)
		.agg(
			mean_delta_rep_rate=("delta_repetition_rate", "mean"),
			mean_abs_delta_rep_rate=("delta_repetition_rate", lambda x: np.mean(np.abs(x))),
			mean_delta_cycle_count=("delta_avg_cycle_count", "mean"),
			n=("head", "count"),
		)
		.dropna(subset=["checkpoint"])
		.sort_values("checkpoint")
	)

	for _, row in summary.iterrows():
		lines.append(
			f"- {row['checkpoint']}: mean Delta rep={row['mean_delta_rep_rate']:.4f}, "
			f"mean |Delta rep|={row['mean_abs_delta_rep_rate']:.4f}, "
			f"mean Delta cycle={row['mean_delta_cycle_count']:.2f}, n={int(row['n'])}"
		)
	lines.append("")
	return lines


def overlap_block(rankings: pd.DataFrame, top_k: int = 10) -> List[str]:
	lines = ["## Natural vs ICL Overlap"]
	natural = rankings[rankings["condition"] == "natural"].head(top_k)
	icl = rankings[rankings["condition"] == "icl"].head(top_k)
	if natural.empty or icl.empty:
		lines.append("- Overlap not computed (one condition missing).")
		lines.append("")
		return lines

	natural_ids = set((int(r.layer), int(r.head)) for r in natural.itertuples())
	icl_ids = set((int(r.layer), int(r.head)) for r in icl.itertuples())
	overlap = natural_ids.intersection(icl_ids)
	jaccard = len(overlap) / len(natural_ids.union(icl_ids)) if natural_ids.union(icl_ids) else 0.0

	lines.append(f"- Top-{top_k} overlap count: {len(overlap)}")
	lines.append(f"- Top-{top_k} Jaccard: {jaccard:.3f}")
	if overlap:
		lines.append("- Shared heads: " + ", ".join([f"L{l}H{h}" for l, h in sorted(overlap)]))
	lines.append("")
	return lines


def condition_difference_block(rankings: pd.DataFrame) -> List[str]:
	lines = ["## Condition-Specific Heads"]
	pivot = rankings.pivot_table(
		index=["layer", "head"],
		columns="condition",
		values="mean_rep_rate_delta",
		aggfunc="mean",
	).reset_index()

	if "natural" not in pivot.columns or "icl" not in pivot.columns:
		lines.append("- Condition difference not available (requires both natural and icl rows).")
		lines.append("")
		return lines

	both = pivot.dropna(subset=["natural", "icl"]).copy()
	if both.empty:
		lines.append("- No heads have data in both conditions.")
		lines.append("")
		return lines

	both["icl_minus_natural"] = both["icl"] - both["natural"]
	if np.allclose(both["icl_minus_natural"].to_numpy(), 0.0):
		lines.append("- ICL-Natural Delta repetition differences are all near zero.")

	top_hi = both.sort_values("icl_minus_natural", ascending=False).head(5)
	top_lo = both.sort_values("icl_minus_natural", ascending=True).head(5)

	lines.append("- Highest ICL-Natural Delta repetition differences:")
	for _, row in top_hi.iterrows():
		lines.append(
			f"  - L{int(row['layer'])}H{int(row['head'])}: natural={row['natural']:.4f}, "
			f"icl={row['icl']:.4f}, diff={row['icl_minus_natural']:.4f}"
		)

	lines.append("- Lowest ICL-Natural Delta repetition differences:")
	for _, row in top_lo.iterrows():
		lines.append(
			f"  - L{int(row['layer'])}H{int(row['head'])}: natural={row['natural']:.4f}, "
			f"icl={row['icl']:.4f}, diff={row['icl_minus_natural']:.4f}"
		)
	lines.append("")
	return lines


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate condition-aware ablation conclusions")
	parser.add_argument("--plots_dir", type=str, default="/home/mmahaut/projects/parrots/plots/ablation_pythia70m")
	args = parser.parse_args()

	plots_dir = Path(args.plots_dir)
	rankings_path = plots_dir / "head_rankings_aggregated.csv"
	combined_path = plots_dir / "combined_ablation_effects.csv"
	if not rankings_path.exists() or not combined_path.exists():
		raise SystemExit(f"Missing input files in {plots_dir}")

	rankings = pd.read_csv(rankings_path)
	combined = pd.read_csv(combined_path)
	conditions = sorted(combined["condition"].dropna().unique().tolist())
	combined_counts = {c: int((combined["condition"] == c).sum()) for c in conditions}
	ranked_counts = {c: int((rankings["condition"] == c).sum()) for c in conditions}

	lines = [
		"# Condition-Aware Causal Ablation Conclusions",
		"",
		"## Run Snapshot",
		f"- Total rows in combined dataset: {len(combined)}",
		f"- Rows per condition: {', '.join([f'{c}={n}' for c, n in combined_counts.items()])}",
		f"- Total ranked layer-head entries: {len(rankings)}",
		f"- Ranked per condition: {', '.join([f'{c}={n}' for c, n in ranked_counts.items()])}",
		f"- Conditions present: {', '.join(conditions) if conditions else 'none'}",
		"",
		"## Causal Sign Convention",
		"- Positive Delta repetition rate (ablated - baseline): the head suppresses repetition in the intact model.",
		"- Negative Delta repetition rate (ablated - baseline): the head promotes repetition in the intact model.",
		"",
	]

	for condition in conditions:
		lines.extend(top_heads_block(rankings, condition, positive=True, top_k=5))
		lines.extend(top_heads_block(rankings, condition, positive=False, top_k=5))

	for condition in conditions:
		lines.extend(checkpoint_block(combined, condition))

	lines.extend(overlap_block(rankings, top_k=10))
	lines.extend(condition_difference_block(rankings))

	lines.extend([
		"## Practical Readout",
		"- Shared top heads across conditions support condition-general repetition control.",
		"- Large ICL-Natural differences support condition-specific head specialization.",
		"- Use top heads for targeted multi-head interventions.",
		"",
	])

	out_file = plots_dir / "CONCLUSIONS.md"
	out_file.write_text("\n".join(lines))
	print(f"Saved: {out_file}")


if __name__ == "__main__":
	main()
