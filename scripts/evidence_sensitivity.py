"""
Sweep the evidence grounding threshold (τ_evidence) and log grounding rejections and quality.

Reports per τ_evidence:
- Final Rel/CRScore and ΔRel vs. seed.
- % of candidates rejected by the grounding guardrail (evidence_ok == False).
- LLM call counts (iterations * num_samples).

Unsupported/contradicted claim rates can be manually flagged in the JSONL outputs (field: unsupported_flag).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language
from core.loop import IterativeRefiner, LoopConfig
from core.scoring import CRScorer
from threshold_refine import choose_editor


def parse_thresholds(raw: Iterable[str]) -> List[float]:
    vals = []
    for item in raw:
        vals.extend([float(x) for x in str(item).split(",") if x.strip() != ""])
    return sorted(set(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep τ_evidence for grounding sensitivity.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT / "CRScore-human_study" / "phase1" / "raw_data.json",
        help="Path to raw data JSON.",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="CRScore similarity threshold.")
    parser.add_argument(
        "--tau-evidence",
        nargs="+",
        default=["0.25,0.35,0.45"],
        help="List/comma-separated τ_evidence values, e.g. --tau-evidence 0.25 0.35 0.45.",
    )
    parser.add_argument("--max-iter", type=int, default=3, help="Iteration budget (N).")
    parser.add_argument("--num-samples", type=int, default=2, help="Candidates per iteration (K).")
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
        default=ROOT / "results" / "evidence_sensitivity",
        help="Directory for outputs (JSON + MD).",
    )
    return parser.parse_args()


def run_for_tau(
    tau_ev: float,
    instances: Sequence[ReviewInstance],
    scorer: CRScorer,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cfg = LoopConfig(
        max_iter=args.max_iter,
        num_samples=args.num_samples,
        tau=args.tau,
        tau_evidence=tau_ev,
        selection=args.selection,
        prompt_style=args.prompt_style,
    )
    editor = choose_editor(args)
    out_path = args.output_dir / f"tau_evidence_{tau_ev:.2f}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total_candidates = 0
    evidence_rejected = 0
    rel_deltas: List[float] = []
    rel_finals: List[float] = []
    rel_seeds: List[float] = []
    llm_calls = 0
    improved = 0

    for inst in instances:
        refiner = IterativeRefiner(scorer=scorer, editor=editor, config=cfg)
        seed_score = scorer.score(inst.pseudo_refs, inst.review)
        res = refiner.run(inst)
        final_score = res["best_score"]
        rel_seeds.append(seed_score.relevance)
        rel_finals.append(final_score["Rel"])
        rel_deltas.append(final_score["Rel"] - seed_score.relevance)
        hist = res["history"]
        cand_count = sum(len(h["candidates"]) for h in hist)
        rej_count = sum(1 for h in hist for c in h["candidates"] if not c.get("evidence_ok", True))
        total_candidates += cand_count
        evidence_rejected += rej_count
        llm_calls += len(hist) * cfg.num_samples
        if final_score["Rel"] > seed_score.relevance:
            improved += 1
        records.append(
            {
                "instance": res["instance"],
                "seed_score": seed_score.to_dict(),
                "final_score": final_score,
                "history": hist,
                "tau_evidence": tau_ev,
                "llm_calls": len(hist) * cfg.num_samples,
                "unsupported_flag": None,  # placeholder for manual check
                "cand_rejections": rej_count,
                "cand_total": cand_count,
            }
        )

    out_path.write_text("\n".join(json.dumps(r) for r in records))
    return {
        "tau_evidence": tau_ev,
        "total": len(instances),
        "improved": improved,
        "avg_seed_rel": float(np.mean(rel_seeds)) if rel_seeds else 0.0,
        "avg_final_rel": float(np.mean(rel_finals)) if rel_finals else 0.0,
        "avg_rel_delta": float(np.mean(rel_deltas)) if rel_deltas else 0.0,
        "median_rel_delta": float(np.median(rel_deltas)) if rel_deltas else 0.0,
        "cand_rejections": evidence_rejected,
        "cand_total": total_candidates,
        "cand_rejection_rate": (evidence_rejected / total_candidates) if total_candidates else 0.0,
        "avg_llm_calls_per_item": (llm_calls / len(instances)) if instances else 0.0,
        "llm_calls_total": llm_calls,
        "output": str(out_path),
    }


def write_md(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    if not summaries:
        return
    lines = [
        "# τ_evidence sensitivity",
        "",
        "| τ_evidence | improved | avg final Rel | avg ΔRel | median ΔRel | cand rejection rate | avg LLM calls/item | output |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['tau_evidence']:.2f} | {row['improved']} / {row['total']} | "
            f"{row['avg_final_rel']:.3f} | {row['avg_rel_delta']:.3f} | {row['median_rel_delta']:.3f} | "
            f"{row['cand_rejection_rate']:.3f} | {row['avg_llm_calls_per_item']:.3f} | {row['output']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    tau_values = parse_thresholds(args.tau_evidence)
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    if args.limit:
        selected = selected[: args.limit]

    scorer = CRScorer(tau=args.tau)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for tau_ev in tau_values:
        summaries.append(run_for_tau(tau_ev, selected, scorer, args))

    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    write_md(args.output_dir / "summary.md", summaries)


if __name__ == "__main__":
    main()
