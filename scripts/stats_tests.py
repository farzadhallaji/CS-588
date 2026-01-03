"""
Statistical testing and effect sizes on per-instance CRScore results.

Supports:
- Within-file paired test (seed_score vs final_score) for threshold/refinement outputs.
- Cross-file paired test on final_score (Rel) matched by instance idx.

Outputs JSON and markdown with:
- mean ΔRel + bootstrap 95% CI
- paired Wilcoxon p-value
- effect sizes: Cohen's d (paired) and Cliff's delta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired stats on CRScore Rel.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--within-file", type=Path, help="JSONL with seed_score/final_score fields for paired comparison.")
    mode.add_argument("--file-a", type=Path, help="JSONL file A with final_score.")
    parser.add_argument("--file-b", type=Path, help="JSONL file B with final_score (required if --file-a is used).")
    parser.add_argument("--label-a", type=str, default="A", help="Label for group A.")
    parser.add_argument("--label-b", type=str, default="B", help="Label for group B.")
    parser.add_argument("--metric", type=str, default="Rel", help="Score key to compare (default: Rel).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "stats_tests")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap samples.")
    return parser.parse_args()


def load_rel_from_file(path: Path, metric: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            inst = rec.get("instance", {}) or {}
            idx = str(inst.get("idx"))
            score = rec.get("final_score") or rec.get("score") or {}
            if idx is None or metric not in score:
                continue
            data[idx] = float(score[metric])
    return data


def load_seed_final(path: Path, metric: str) -> Tuple[List[float], List[float]]:
    seeds, finals = [], []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            seed = rec.get("seed_score") or {}
            final = rec.get("final_score") or rec.get("score") or {}
            if metric in seed and metric in final:
                seeds.append(float(seed[metric]))
                finals.append(float(final[metric]))
    return seeds, finals


def bootstrap_ci(diffs: np.ndarray, n: int = 5000, alpha: float = 0.05) -> Tuple[float, float]:
    if diffs.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed=42)
    samples = rng.choice(diffs, size=(n, diffs.size), replace=True).mean(axis=1)
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return lower, upper


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # For paired, operate on differences (b - a)
    diffs = b - a
    n = diffs.size
    if n == 0:
        return 0.0
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    return float((pos - neg) / (n))


def paired_stats(a: np.ndarray, b: np.ndarray, n_boot: int, label_a: str, label_b: str) -> Dict[str, object]:
    diffs = b - a
    mean_diff = float(np.mean(diffs)) if diffs.size else 0.0
    median_diff = float(np.median(diffs)) if diffs.size else 0.0
    ci_low, ci_high = bootstrap_ci(diffs, n_boot)
    try:
        wilcoxon_p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    sd_diff = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    cohen_d = mean_diff / sd_diff if sd_diff > 0 else 0.0
    delta = cliffs_delta(a, b)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n": int(diffs.size),
        "mean_delta": mean_diff,
        "median_delta": median_diff,
        "ci_95": [ci_low, ci_high],
        "wilcoxon_p": wilcoxon_p,
        "cohen_d": cohen_d,
        "cliffs_delta": delta,
    }


def write_md(path: Path, stats_row: Dict[str, object]) -> None:
    lines = [
        "# Paired stats",
        "",
        "| n | mean Δ | median Δ | 95% CI | Wilcoxon p | Cohen's d | Cliff's δ |",
        "|---|---|---|---|---|---|---|",
        f"| {stats_row['n']} | {stats_row['mean_delta']:.4f} | {stats_row['median_delta']:.4f} | "
        f"[{stats_row['ci_95'][0]:.4f}, {stats_row['ci_95'][1]:.4f}] | {stats_row['wilcoxon_p']:.4g} | "
        f"{stats_row['cohen_d']:.4f} | {stats_row['cliffs_delta']:.4f} |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.within_file:
        a_list, b_list = load_seed_final(args.within_file, args.metric)
        a = np.array(a_list)
        b = np.array(b_list)
        stats_row = paired_stats(a, b, args.n_bootstrap, label_a="seed", label_b="final")
        base = args.output_dir / f"{args.within_file.stem}_within"
    else:
        if not args.file_a or not args.file_b:
            raise ValueError("Provide --file-a and --file-b for cross-file comparison.")
        a_dict = load_rel_from_file(args.file_a, args.metric)
        b_dict = load_rel_from_file(args.file_b, args.metric)
        common = sorted(set(a_dict.keys()) & set(b_dict.keys()))
        a = np.array([a_dict[k] for k in common])
        b = np.array([b_dict[k] for k in common])
        stats_row = paired_stats(a, b, args.n_bootstrap, label_a=args.label_a, label_b=args.label_b)
        base = args.output_dir / f"{args.file_a.stem}_vs_{args.file_b.stem}"

    (base.with_suffix(".json")).write_text(json.dumps(stats_row, indent=2))
    write_md(base.with_suffix(".md"), stats_row)


if __name__ == "__main__":
    main()
