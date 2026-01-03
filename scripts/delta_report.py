"""
Generate instance-level ΔCRScore tables and summary stats from existing outputs.

Inputs:
- JSONL produced by run.py (iterative loop) or threshold_refine.py (seed/final scores).

Outputs:
- Markdown table of up to N instances with seed Con/Comp/Rel, final Con/Comp/Rel, ΔRel, best iteration (if available).
- Summary stats: count, mean/median ΔRel, IQR, min/max.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize instance-level CRScore deltas.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file with seed_score/final_score fields.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output file.")
    parser.add_argument("--limit", type=int, default=25, help="Max rows to include in the table.")
    parser.add_argument(
        "--sort-by",
        choices=["delta_desc", "delta_asc", "idx"],
        default="delta_desc",
        help="Sorting for table rows.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            inst = rec.get("instance", {}) or {}
            idx = inst.get("idx")
            history = rec.get("history") or []
            # Seed: prefer explicit seed_score; otherwise fall back to the first history score if present.
            seed = rec.get("seed_score") or rec.get("score") or (history[0].get("score") if history else {}) or {}
            # Final: prefer explicit final_score; otherwise fall back to best_score (run.py) or score.
            final = rec.get("final_score") or rec.get("best_score") or rec.get("score") or {}
            records.append({"idx": idx, "seed": seed, "final": final, "history": history})
    return records


def find_best_iter(history: List[Dict[str, object]], best_rel: float) -> Optional[int]:
    if not history:
        return None
    for h in history:
        sel = h.get("selected") or {}
        sel_score = sel.get("score") or {}
        rel = sel_score.get("Rel")
        if rel is not None and abs(rel - best_rel) < 1e-6:
            return h.get("iter")
    return history[-1].get("iter")


def build_table(records: List[Dict[str, object]], limit: int, sort_by: str) -> str:
    rows = []
    for r in records:
        seed = r["seed"]
        final = r["final"]
        seed_rel = seed.get("Rel", 0.0)
        final_rel = final.get("Rel", 0.0)
        delta = final_rel - seed_rel
        best_iter = find_best_iter(r.get("history", []), final_rel)
        rows.append(
            {
                "idx": r["idx"],
                "seed": seed,
                "final": final,
                "delta": delta,
                "best_iter": best_iter,
            }
        )
    if sort_by == "delta_desc":
        rows.sort(key=lambda x: x["delta"], reverse=True)
    elif sort_by == "delta_asc":
        rows.sort(key=lambda x: x["delta"])
    else:
        rows.sort(key=lambda x: (x["idx"] is None, x["idx"]))

    header = [
        "| idx | seed Con | seed Comp | seed Rel | final Con | final Comp | final Rel | ΔRel | best iter |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    body = []
    for row in rows[:limit]:
        body.append(
            f"| {row['idx']} | {row['seed'].get('Con', 0):.3f} | {row['seed'].get('Comp', 0):.3f} | {row['seed'].get('Rel', 0):.3f} | "
            f"{row['final'].get('Con', 0):.3f} | {row['final'].get('Comp', 0):.3f} | {row['final'].get('Rel', 0):.3f} | "
            f"{row['delta']:.3f} | {row['best_iter'] if row['best_iter'] is not None else '-'} |"
        )
    return "\n".join(header + body)


def summary_stats(records: List[Dict[str, object]]) -> Dict[str, float]:
    deltas = np.array([r["final"].get("Rel", 0.0) - r["seed"].get("Rel", 0.0) for r in records])
    if deltas.size == 0:
        return {"n": 0}
    return {
        "n": int(deltas.size),
        "mean": float(np.mean(deltas)),
        "median": float(np.median(deltas)),
        "p25": float(np.percentile(deltas, 25)),
        "p75": float(np.percentile(deltas, 75)),
        "min": float(np.min(deltas)),
        "max": float(np.max(deltas)),
    }


def write_md(path: Path, table_md: str, stats: Dict[str, float]) -> None:
    lines = [
        "# Instance-level ΔCRScore",
        "",
        "## Summary",
        f"- n={stats.get('n', 0)}, mean ΔRel={stats.get('mean', 0):.3f}, median={stats.get('median', 0):.3f}, "
        f"IQR=[{stats.get('p25', 0):.3f}, {stats.get('p75', 0):.3f}], min={stats.get('min', 0):.3f}, max={stats.get('max', 0):.3f}",
        "",
        "## Examples",
        table_md,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    stats = summary_stats(records)
    table_md = build_table(records, limit=args.limit, sort_by=args.sort_by)
    write_md(args.output, table_md, stats)


if __name__ == "__main__":
    main()
