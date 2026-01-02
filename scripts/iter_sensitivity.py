"""
Sweep iteration budget (N = max_iter) with fixed K candidates per iteration.

Reports per N:
- Final Rel/CRScore and ΔRel vs. seed.
- Avg iterations used, LLM calls per item (iterations * K).
- Whether gains saturate across N values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language
from core.loop import IterativeRefiner, LoopConfig
from core.scoring import CRScorer
from threshold_refine import choose_editor


def parse_iters(raw: Iterable[str]) -> List[int]:
    vals: List[int] = []
    for item in raw:
        vals.extend([int(x) for x in str(item).split(",") if x.strip() != ""])
    return sorted(set(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteration budget (N) sensitivity with fixed K.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT / "CRScore-human_study" / "phase1" / "raw_data.json",
        help="Path to raw data JSON.",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="CRScore similarity threshold.")
    parser.add_argument("--tau-evidence", type=float, default=LoopConfig.tau_evidence, help="Evidence similarity threshold.")
    parser.add_argument(
        "--iters",
        nargs="+",
        default=["1,2,3"],
        help="List/comma-separated iteration budgets N, e.g. --iters 1 2 3.",
    )
    parser.add_argument("--num-samples", type=int, default=2, help="Fixed K candidates per iteration.")
    parser.add_argument("--prompt-style", choices=["loop", "k1", "rewrite", "no-evidence", "no-selection"], default="loop")
    parser.add_argument("--selection", choices=["crscore", "random", "shortest"], default="crscore")
    parser.add_argument("--model-type", choices=["ollama", "hf-local"], default="ollama")
    parser.add_argument("--model-name", type=str, default="llama3:8b-instruct-q4_0", help="Ollama model name or HF path.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "iter_sensitivity",
        help="Directory for outputs (JSON + MD).",
    )
    return parser.parse_args()


def run_for_iters(
    n_iter: int,
    instances: Sequence[ReviewInstance],
    scorer: CRScorer,
    args: argparse.Namespace,
) -> dict:
    cfg = LoopConfig(
        max_iter=n_iter,
        num_samples=args.num_samples,
        tau=args.tau,
        tau_evidence=args.tau_evidence,
        selection=args.selection,
        prompt_style=args.prompt_style,
    )
    editor = choose_editor(args)
    out_path = args.output_dir / f"iters_{n_iter}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    rel_seeds: List[float] = []
    rel_finals: List[float] = []
    rel_deltas: List[float] = []
    iter_counts: List[int] = []
    llm_calls = 0
    improved = 0

    for inst in instances:
        refiner = IterativeRefiner(scorer=scorer, editor=editor, config=cfg)
        seed_score = scorer.score(inst.pseudo_refs, inst.review)
        res = refiner.run(inst)
        final_score = res["best_score"]
        hist = res["history"]
        rel_seeds.append(seed_score.relevance)
        rel_finals.append(final_score["Rel"])
        rel_deltas.append(final_score["Rel"] - seed_score.relevance)
        iter_counts.append(len(hist))
        llm_calls += len(hist) * cfg.num_samples
        if final_score["Rel"] > seed_score.relevance:
            improved += 1
        records.append(
            {
                "instance": res["instance"],
                "seed_score": seed_score.to_dict(),
                "final_score": final_score,
                "history": hist,
                "iters_used": len(hist),
                "llm_calls": len(hist) * cfg.num_samples,
            }
        )

    out_path.write_text("\n".join(json.dumps(r) for r in records))
    return {
        "iters": n_iter,
        "total": len(instances),
        "improved": improved,
        "avg_final_rel": float(np.mean(rel_finals)) if rel_finals else 0.0,
        "avg_rel_delta": float(np.mean(rel_deltas)) if rel_deltas else 0.0,
        "median_rel_delta": float(np.median(rel_deltas)) if rel_deltas else 0.0,
        "avg_iters_used": float(np.mean(iter_counts)) if iter_counts else 0.0,
        "avg_llm_calls_per_item": (llm_calls / len(instances)) if instances else 0.0,
        "llm_calls_total": llm_calls,
        "output": str(out_path),
    }


def write_md(path: Path, summaries: Sequence[dict]) -> None:
    if not summaries:
        return
    lines = [
        "# Iteration budget sensitivity",
        "",
        "| N (max_iter) | improved | avg final Rel | avg ΔRel | median ΔRel | avg iters used | avg LLM calls/item | output |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['iters']} | {row['improved']} / {row['total']} | {row['avg_final_rel']:.3f} | "
            f"{row['avg_rel_delta']:.3f} | {row['median_rel_delta']:.3f} | {row['avg_iters_used']:.3f} | "
            f"{row['avg_llm_calls_per_item']:.3f} | {row['output']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    iter_values = parse_iters(args.iters)
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    if args.limit:
        selected = selected[: args.limit]

    scorer = CRScorer(tau=args.tau)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for n_iter in iter_values:
        summaries.append(run_for_iters(n_iter, selected, scorer, args))

    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    write_md(args.output_dir / "summary.md", summaries)


if __name__ == "__main__":
    main()
