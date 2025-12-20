"""
Comprehensive results analysis and plotting for the ProposedApproach pipeline.

Reads summary JSON files (and selected JSONL outputs) from the results directory,
aggregates Con/Comp/Rel metrics, computes improvement rates for threshold runs,
and writes CSV/Markdown tables plus publication-ready plots.

Usage (from repo root):
  python ProposedApproach/scripts/analyze_results.py \
    --results-dir ProposedApproach/results \
    --out-dir ProposedApproach/results/analysis

Notes:
- Expects summary files named like:
    loop_summary.json, single_edit_summary.json, single_rewrite_summary.json,
    no_selection_summary.json, no_evidence_summary.json, rewrite_loop_summary.json,
    baseline_summary.json,
    proposal_v1_<model>_summary.json,
    threshold_<prompt>_<model>_summary.json.
- For threshold improvement stats, it reads the matching JSONL:
    threshold_<prompt>_<model>.jsonl (expects "seed_score", "final_score", "improved").
- Missing files are skipped gracefully.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class Metrics:
    con: float
    comp: float
    rel: float
    n: int


def load_summary(path: Path) -> Optional[Metrics]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    overall = data.get("overall", {})
    try:
        return Metrics(
            con=float(overall.get("Con", 0.0)),
            comp=float(overall.get("Comp", 0.0)),
            rel=float(overall.get("Rel", 0.0)),
            n=int(overall.get("N", 0)),
        )
    except Exception:
        return None


def load_threshold_improvement(jsonl_path: Path) -> Tuple[int, int, float]:
    """Return (total, improved, avg_rel_gain) for a threshold JSONL file."""
    if not jsonl_path.exists():
        return 0, 0, 0.0
    total = 0
    improved = 0
    rel_gains: List[float] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        if rec.get("improved"):
            improved += 1
        seed = rec.get("seed_score", {})
        final = rec.get("final_score", {})
        seed_rel = float(seed.get("Rel", 0.0))
        final_rel = float(final.get("Rel", 0.0))
        rel_gains.append(final_rel - seed_rel)
    avg_gain = sum(rel_gains) / len(rel_gains) if rel_gains else 0.0
    return total, improved, avg_gain


def scan_results(results_dir: Path):
    main_systems = {}
    proposal = {}
    threshold = {}

    for path in results_dir.glob("*_summary.json"):
        name = path.name
        # Core systems and ablations
        if name in {
            "loop_summary.json",
            "single_edit_summary.json",
            "single_rewrite_summary.json",
            "no_selection_summary.json",
            "no_evidence_summary.json",
            "rewrite_loop_summary.json",
            "baseline_summary.json",
        }:
            key = name.replace("_summary.json", "")
            main_systems[key] = load_summary(path)
            continue

        # Proposal v1 sweeps
        m_prop = re.match(r"proposal_v1_(.+)_summary\.json", name)
        if m_prop:
            model = m_prop.group(1)
            proposal[model] = load_summary(path)
            continue

        # Threshold sweeps
        m_thr = re.match(r"threshold_(.+)_(.+)_summary\.json", name)
        if m_thr:
            prompt = m_thr.group(1)
            model = m_thr.group(2)
            threshold[(prompt, model)] = {
                "metrics": load_summary(path),
                "jsonl": results_dir / f"threshold_{prompt}_{model}.jsonl",
            }
            continue

    return main_systems, proposal, threshold


def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)


def write_table_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    path.write_text("\n".join(lines))


def write_table_md(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    headers = list(rows[0].keys())
    header_line = " | ".join(headers)
    sep_line = " | ".join(["---"] * len(headers))
    lines = [header_line, sep_line]
    for r in rows:
        lines.append(" | ".join(str(r.get(h, "")) for h in headers))
    path.write_text("\n".join(lines))


def plot_bar(path: Path, title: str, entries: List[Tuple[str, float]], ylabel: str = "Rel"):
    if not entries:
        return
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    bars = plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_threshold_grid(path: Path, entries: List[Tuple[str, str, float]]):
    """Heatmap-like bar plot: x=models, grouped by prompt."""
    if not entries:
        return
    # group by prompt
    grouped = defaultdict(list)
    for prompt, model, rel in entries:
        grouped[prompt].append((model, rel))
    prompts = sorted(grouped.keys())
    models = sorted({m for _, m, _ in entries})
    width = 0.18
    x = range(len(models))
    plt.figure(figsize=(max(7, len(models) * 1.0), 4.8))
    for i, prompt in enumerate(prompts):
        vals = []
        for m in models:
            rel = next((r for mod, r in grouped[prompt] if mod == m), 0.0)
            vals.append(rel)
        offsets = [xi + (i - (len(prompts) - 1) / 2) * width for xi in x]
        plt.bar(offsets, vals, width=width, label=prompt)
    plt.xticks(list(x), models, rotation=30, ha="right")
    plt.ylabel("Rel")
    plt.title("Threshold refinement: Rel by model × prompt")
    plt.legend(title="Prompt", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ProposedApproach results.")
    parser.add_argument("--results-dir", type=Path, default=Path("ProposedApproach/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("ProposedApproach/results/analysis"))
    args = parser.parse_args()

    ensure_out(args.out_dir)
    main_systems, proposal, threshold = scan_results(args.results_dir)

    # Main systems table
    main_rows = []
    for key, met in main_systems.items():
        if not met:
            continue
        label = key.replace("_", " ")
        main_rows.append({"system": label, "Rel": f"{met.rel:.3f}", "Con": f"{met.con:.3f}", "Comp": f"{met.comp:.3f}", "N": met.n})
    main_rows.sort(key=lambda r: r["system"])
    write_table_csv(args.out_dir / "main_systems.csv", main_rows)
    write_table_md(args.out_dir / "main_systems.md", main_rows)
    plot_bar(args.out_dir / "plots" / "main_systems_rel.png", "Core systems (Rel)", [(r["system"], float(r["Rel"])) for r in main_rows])

    # Proposal sweep table
    prop_rows = []
    for model, met in proposal.items():
        if not met:
            continue
        prop_rows.append({"model": model, "Rel": f"{met.rel:.3f}", "Con": f"{met.con:.3f}", "Comp": f"{met.comp:.3f}", "N": met.n})
    prop_rows.sort(key=lambda r: r["model"])
    write_table_csv(args.out_dir / "proposal_v1.csv", prop_rows)
    write_table_md(args.out_dir / "proposal_v1.md", prop_rows)
    plot_bar(args.out_dir / "plots" / "proposal_v1_rel.png", "Proposal v1 baseline (Rel)", [(r["model"], float(r["Rel"])) for r in prop_rows])

    # Threshold sweep table with improvement rates
    thr_rows = []
    thr_plot_entries = []
    thr_improve_rows = []
    for (prompt, model), info in threshold.items():
        met = info.get("metrics")
        if not met:
            continue
        total, improved, avg_gain = load_threshold_improvement(info["jsonl"])
        thr_rows.append(
            {
                "prompt": prompt,
                "model": model,
                "Rel": f"{met.rel:.3f}",
                "Con": f"{met.con:.3f}",
                "Comp": f"{met.comp:.3f}",
                "N": met.n,
                "improvement_rate": f"{(improved / total):.3f}" if total else "",
                "avg_rel_gain": f"{avg_gain:.3f}" if total else "",
            }
        )
        thr_plot_entries.append((prompt, model, met.rel))
        thr_improve_rows.append({"prompt": prompt, "model": model, "total": total, "improved": improved, "improvement_rate": f"{(improved/total):.3f}" if total else ""})

    thr_rows.sort(key=lambda r: (r["prompt"], r["model"]))
    write_table_csv(args.out_dir / "threshold_summary.csv", thr_rows)
    write_table_md(args.out_dir / "threshold_summary.md", thr_rows)
    write_table_csv(args.out_dir / "threshold_improvement.csv", thr_improve_rows)
    write_table_md(args.out_dir / "threshold_improvement.md", thr_improve_rows)
    plot_threshold_grid(args.out_dir / "plots" / "threshold_rel.png", thr_plot_entries)

    print(f"Wrote analysis tables/plots to {args.out_dir}")


if __name__ == "__main__":
    main()
