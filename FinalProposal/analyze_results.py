#!/usr/bin/env python3
"""
Quick analysis/plotting helper for FinalProposal results.

Reads all summary JSONs and robustness CSVs under results/, aggregates the key
metrics, and emits tables + plots suitable for paper-ready figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"


def load_summaries(summaries_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(summaries_dir.glob("summary_*.json")):
        with path.open() as fh:
            data = json.load(fh)
        overall = data.get("overall", {})
        if not overall:
            continue
        run = path.stem.replace("summary_", "", 1)
        rows.append(
            {
                "run": run,
                "Rel": overall.get("Rel"),
                "Con": overall.get("Con"),
                "Comp": overall.get("Comp"),
                "N": overall.get("N"),
            }
        )
    return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)


def add_deltas(df: pd.DataFrame, base_run: str) -> Tuple[pd.DataFrame, pd.Series | None]:
    if df.empty or base_run not in df["run"].values:
        return df, None
    base_row = df.loc[df["run"] == base_run].iloc[0]
    delta_df = df.copy()
    for metric in ("Rel", "Con", "Comp"):
        delta_df[f"delta_{metric}"] = delta_df[metric] - base_row[metric]
    return delta_df, base_row


def load_robustness(robust_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(robust_dir.glob("robust_*.csv")):
        run = path.stem.replace("robust_", "", 1)
        df = pd.read_csv(path)
        df["run"] = run
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_overall(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = df.set_index("run")[["Rel", "Con", "Comp"]]
    metrics.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylabel("Score")
    ax.set_title("FinalProposal overall metrics")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_deltas(delta_df: pd.DataFrame, out_path: Path) -> None:
    if delta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = delta_df.set_index("run")[["delta_Rel", "delta_Con", "delta_Comp"]]
    metrics.plot(kind="bar", ax=ax, width=0.85, color=["#1b9e77", "#7570b3", "#d95f02"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta vs base")
    ax.set_title("Metric deltas vs base run")
    ax.legend(title="Delta metric")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_robustness(robust_df: pd.DataFrame, out_path: Path) -> None:
    if robust_df.empty:
        return
    # Focus on comp_delta_mean as a single robustness indicator.
    pivot = robust_df.pivot_table(
        index="scenario", columns="run", values="comp_delta_mean", aggfunc="mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Robustness: Comp delta vs base scenario")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Comp delta mean")
    # Annotate values to ease reading.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_report(
    out_dir: Path,
    df: pd.DataFrame,
    delta_df: pd.DataFrame,
    base_row: pd.Series | None,
    robust_df: pd.DataFrame,
) -> None:
    report_path = out_dir / "report.md"
    lines = []
    lines.append("# FinalProposal results summary")
    lines.append("")
    if not df.empty:
        lines.append("## Overall (overall split, per run)")
        lines.append("")
        lines.append(df.to_string(index=False))
        lines.append("")
        if base_row is not None:
            lines.append(f"Base run: {base_row.to_dict()}")
    if not delta_df.empty:
        lines.append("")
        lines.append("## Deltas vs base")
        lines.append("")
        lines.append(delta_df.to_string(index=False))
    if not robust_df.empty:
        lines.append("")
        lines.append("## Robustness (comp_delta_mean per scenario/run)")
        pivot = robust_df.pivot_table(
            index="scenario", columns="run", values="comp_delta_mean", aggfunc="mean"
        ).sort_index()
        lines.append("")
        lines.append(pivot.to_string())
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FinalProposal results and plot.")
    parser.add_argument(
        "--base-run",
        default="base__reward_default",
        help="Run id to use as baseline for deltas (matches summary_* suffix).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR / "analysis",
        help="Directory to write aggregated tables and plots.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(SUMMARIES_DIR)
    delta_df, base_row = add_deltas(summaries, args.base_run)
    robustness = load_robustness(ROBUSTNESS_DIR)

    summaries.to_csv(args.out_dir / "summaries_overall.csv", index=False)
    if not delta_df.empty and base_row is not None:
        delta_df.to_csv(args.out_dir / "summaries_with_deltas.csv", index=False)
    if not robustness.empty:
        robustness.to_csv(args.out_dir / "robustness_all.csv", index=False)

    plot_overall(summaries, args.out_dir / "overall_metrics.png")
    if base_row is not None and not delta_df.empty:
        plot_deltas(delta_df, args.out_dir / "overall_deltas_vs_base.png")
    if not robustness.empty:
        plot_robustness(robustness, args.out_dir / "robustness_comp_delta.png")

    write_report(args.out_dir, summaries, delta_df, base_row, robustness)
    print(f"Wrote analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
