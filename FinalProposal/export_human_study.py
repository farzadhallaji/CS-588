"""
Export paired system outputs for lightweight human evaluation.
Creates a CSV with overlapping items so human_eval.py can compute paired deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple
import sys

from . import PROPOSED_ROOT

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

from core.data import build_instances, load_raw_data  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paired reviews for human annotation.")
    parser.add_argument("--system-a", type=Path, required=True, help="JSONL outputs for system A.")
    parser.add_argument("--system-b", type=Path, required=True, help="JSONL outputs for system B.")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "human_export.csv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_outputs(path: Path) -> Dict[int, dict]:
    outputs: Dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            inst = rec.get("instance", {}) or {}
            idx = inst.get("idx")
            if idx is None:
                continue
            outputs[int(idx)] = rec
    return outputs


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    inst_map = {inst.idx: inst for inst in build_instances(load_raw_data(args.raw_data))}
    out_a = load_outputs(args.system_a)
    out_b = load_outputs(args.system_b)

    overlap_idxs = sorted(set(out_a.keys()) & set(out_b.keys()))
    if args.limit:
        overlap_idxs = overlap_idxs[: args.limit]
    if not overlap_idxs:
        raise SystemExit("No overlapping indices found between system A and B outputs.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        fieldnames = [
            "lang",
            "index",
            "seed_review",
            "claims",
            "diff",
            "system_A",
            "system_B",
            "review_A",
            "review_B",
            "Con (P) A",
            "Comp (R) A",
            "Rel (F) A",
            "Con (P) B",
            "Comp (R) B",
            "Rel (F) B",
            "preferred (A/B)",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx in overlap_idxs:
            inst = inst_map.get(idx)
            if inst is None:
                continue
            rec_a = out_a[idx]
            rec_b = out_b[idx]
            review_a = rec_a.get("best_review") or rec_a.get("seed_review", "")
            review_b = rec_b.get("best_review") or rec_b.get("seed_review", "")

            # Randomize assignment to A/B to mitigate bias.
            if rng.random() < 0.5:
                system_a = rec_a.get("system", "system_a")
                system_b = rec_b.get("system", "system_b")
            else:
                system_a, system_b = rec_b.get("system", "system_b"), rec_a.get("system", "system_a")
                review_a, review_b = review_b, review_a

            writer.writerow(
                {
                    "lang": inst.lang,
                    "index": idx,
                    "seed_review": inst.review,
                    "claims": "\n".join(inst.pseudo_refs),
                    "diff": inst.patch,
                    "system_A": system_a,
                    "system_B": system_b,
                    "review_A": review_a,
                    "review_B": review_b,
                    "Con (P) A": "",
                    "Comp (R) A": "",
                    "Rel (F) A": "",
                    "Con (P) B": "",
                    "Comp (R) B": "",
                    "Rel (F) B": "",
                    "preferred (A/B)": "",
                }
            )

    print(f"Wrote paired export to {args.output}")


if __name__ == "__main__":
    main()
