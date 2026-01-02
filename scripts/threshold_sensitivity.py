"""
Sensitivity utilities for the quality gate threshold (θ).

Capabilities:
- Score seed reviews once, then report how many fall below each θ value (coverage), overall and per language.
- Optionally run threshold-gated refinement for each θ and log outcomes (seed vs. refined scores, improvement flag).

This is intended to address reviewer requests for θ sweeps (e.g., 0.5 vs. 0.6 vs. 0.7) without duplicating data prep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language
from core.scoring import CRScorer, ScoreResult
from core.loop import LoopConfig
from threshold_refine import PROMPTS, build_prompt, choose_editor


def parse_thresholds(raw: Iterable[str]) -> List[float]:
    vals = []
    for item in raw:
        vals.extend([float(x) for x in str(item).split(",") if x.strip() != ""])
    return sorted(set(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run θ sweeps for the CRScore quality gate.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT / "CRScore-human_study" / "phase1" / "raw_data.json",
        help="Path to raw data JSON (default: phase1 human study).",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="CRScore similarity threshold.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=["0.4,0.5,0.6,0.7,0.8"],
        help="List/comma-separated θ values to sweep, e.g. --thresholds 0.5 0.6 0.7 or --thresholds 0.45,0.6,0.75.",
    )
    parser.add_argument("--prompt-variant", choices=list(PROMPTS.keys()), default="default")
    parser.add_argument("--model-type", choices=["ollama", "hf-local"], default="ollama")
    parser.add_argument("--model-name", type=str, default="llama3:8b-instruct-q4_0", help="Ollama model name or HF path.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--run-refine",
        action="store_true",
        help="If set, perform gated refinement for each θ and write per-threshold JSONL outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "threshold_sensitivity",
        help="Directory for coverage summary and optional refinement outputs.",
    )
    return parser.parse_args()


def score_instances(instances: Sequence[ReviewInstance], scorer: CRScorer) -> List[ScoreResult]:
    return [scorer.score(inst.pseudo_refs, inst.review) for inst in instances]


def coverage_summary(
    instances: Sequence[ReviewInstance], scores: Sequence[ScoreResult], thresholds: Sequence[float]
) -> List[Dict[str, object]]:
    summary = []
    for theta in thresholds:
        gated = [(inst, sc) for inst, sc in zip(instances, scores) if sc.relevance < theta]
        per_lang: Dict[str, int] = {}
        for inst, _ in gated:
            per_lang[inst.lang] = per_lang.get(inst.lang, 0) + 1
        summary.append(
            {
                "theta": theta,
                "total": len(instances),
                "gated": len(gated),
                "gated_pct": 0.0 if not instances else len(gated) / len(instances),
                "avg_seed_rel": float(np.mean([sc.relevance for sc in scores])) if scores else 0.0,
                "avg_gated_rel": float(np.mean([sc.relevance for _, sc in gated])) if gated else 0.0,
                "per_lang": per_lang,
            }
        )
    return summary


def run_refinement(
    instances: Sequence[ReviewInstance],
    scores: Sequence[ScoreResult],
    thresholds: Sequence[float],
    scorer: CRScorer,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    editor = choose_editor(args)
    summaries = []
    for theta in thresholds:
        out_path = args.output_dir / f"threshold_{theta:.2f}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gated = 0
        improved = 0
        rel_deltas: List[float] = []
        llm_calls = 0
        iter_counts: List[int] = []
        with out_path.open("w") as f:
            for inst, seed_score in zip(instances, scores):
                best_review = inst.review
                final_score = seed_score
                did_refine = seed_score.relevance < theta
                iter_used = 0
                if did_refine:
                    gated += 1
                    iter_used = 1  # single-pass refine under gate
                    prompt = build_prompt(inst.review, inst.pseudo_refs, args.prompt_variant)
                    cand = editor.propose(
                        current_review=inst.review,
                        uncovered=[],
                        offending=[],
                        evidence={},
                        prompt=prompt,
                        num_samples=1,
                    )[0]
                    best_review = cand.strip() or inst.review
                    final_score = scorer.score(inst.pseudo_refs, best_review)
                    rel_deltas.append(final_score.relevance - seed_score.relevance)
                    llm_calls += 1
                    if final_score.relevance > seed_score.relevance:
                        improved += 1
                iter_counts.append(iter_used)
                record = {
                    "instance": {"idx": inst.idx, "lang": inst.lang, "meta": inst.meta},
                    "seed_score": seed_score.to_dict(),
                    "final_score": final_score.to_dict(),
                    "seed_review": inst.review,
                    "best_review": best_review,
                    "threshold": theta,
                    "did_refine": did_refine,
                    "prompt_variant": args.prompt_variant,
                    "model_type": args.model_type,
                    "model_name": args.model_name,
                }
                f.write(json.dumps(record) + "\n")
        summaries.append(
            {
                "theta": theta,
                "gated": gated,
                "gated_pct": 0.0 if not instances else gated / len(instances),
                "improved": improved,
                "avg_rel_delta": float(np.mean(rel_deltas)) if rel_deltas else 0.0,
                "median_rel_delta": float(np.median(rel_deltas)) if rel_deltas else 0.0,
                "avg_iterations_per_item": float(np.mean(iter_counts)) if iter_counts else 0.0,
                "avg_llm_calls_per_item": (llm_calls / len(instances)) if instances else 0.0,
                "llm_calls_total": llm_calls,
                "output": str(out_path),
            }
        )
    return summaries


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_md_coverage(path: Path, coverage: Sequence[Dict[str, object]]) -> None:
    lines = ["# θ coverage summary", "", "| θ | gated | gated % | avg seed Rel | avg gated Rel | per-lang gated |", "|---|---|---|---|---|---|"]
    for row in coverage:
        per_lang = ", ".join(f"{k}:{v}" for k, v in sorted(row["per_lang"].items())) if row["per_lang"] else "-"
        lines.append(
            f"| {row['theta']:.2f} | {row['gated']} / {row['total']} | {row['gated_pct']:.3f} | "
            f"{row['avg_seed_rel']:.3f} | {row['avg_gated_rel']:.3f} | {per_lang} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_md_refine(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    if not summaries:
        return
    lines = [
        "# θ refinement summary",
        "",
        "| θ | gated % | improved | avg ΔRel | median ΔRel | avg iters | avg LLM calls/item | output |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['theta']:.2f} | {row['gated_pct']:.3f} | {row['improved']} / {row['gated']} | "
            f"{row['avg_rel_delta']:.3f} | {row['median_rel_delta']:.3f} | "
            f"{row['avg_iterations_per_item']:.3f} | {row['avg_llm_calls_per_item']:.3f} | {row['output']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    if args.limit:
        selected = selected[: args.limit]

    scorer = CRScorer(tau=args.tau)
    scores = score_instances(selected, scorer)

    coverage = coverage_summary(selected, scores, thresholds)
    write_json(args.output_dir / "coverage.json", coverage)
    write_md_coverage(args.output_dir / "coverage.md", coverage)

    refine_summaries: List[Dict[str, object]] = []
    if args.run_refine:
        refine_summaries = run_refinement(selected, scores, thresholds, scorer, args)
        write_json(args.output_dir / "refine_summary.json", refine_summaries)
        write_md_refine(args.output_dir / "refine_summary.md", refine_summaries)


if __name__ == "__main__":
    main()
