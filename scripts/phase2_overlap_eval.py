"""
Phase2 overlap analysis: find ids rated by humans for a target system, subset the
phase1 dataset to those ids, and optionally evaluate provided system outputs
against the human scores.

Typical use:
  python ProposedApproach/scripts/phase2_overlap_eval.py \
    --raw-data ProposedApproach/../CRScore/human_study/phase1/raw_data.json \
    --phase2-dir ProposedApproach/../CRScore/human_study/phase2 \
    --target-system gpt3.5_pred \
    --outputs ProposedApproach/results/threshold_default_llama3_8b-instruct-q4_0.jsonl \
    --summary-out ProposedApproach/results/phase2_overlap_summary.json

Outputs:
  - Prints overlap stats (how many ids are rated for baseline vs target).
  - If --summary-out is set, writes a JSON summary with paired deltas (human)
    and CRScore vs human correlations on the overlap set when outputs are given.

Notes:
  - Baseline system defaults to "msg" (human seed reviews).
  - Overlap is defined as ids that have human ratings for BOTH baseline and the
    target system.
  - If your outputs file does not cover these ids, correlation counts may be 0.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language  # type: ignore  # noqa: E402
from core.scoring import CRScorer  # type: ignore  # noqa: E402
from human_eval import load_phase2_scores, read_outputs  # type: ignore  # noqa: E402


def parse_lang_from_name(path: Path) -> str:
    stem = path.name.split("_")[0]
    if stem not in {"py", "java", "js"}:
        raise ValueError(f"Could not infer language from {path.name}")
    return stem


def find_overlap_ids(phase2_dir: Path, pattern: str, baseline: str, target: str) -> Dict[str, List[int]]:
    files = sorted(phase2_dir.glob(pattern))
    ratings = load_phase2_scores(files)  # key: (lang, idx, system)
    overlap: Dict[str, List[int]] = {"java": [], "js": [], "py": []}
    for (lang, idx, system) in ratings:
        if system != baseline:
            continue
        # baseline exists; check target
        if (lang, idx, target) in ratings:
            overlap.setdefault(lang, []).append(idx)
    return overlap


def filter_instances(instances: List[ReviewInstance], overlap: Dict[str, List[int]]) -> List[ReviewInstance]:
    keep = []
    idx_sets = {lang: set(idxs) for lang, idxs in overlap.items()}
    for inst in instances:
        if inst.idx in idx_sets.get(inst.lang, set()):
            keep.append(inst)
    return keep


def summarize(
    overlap: Dict[str, List[int]],
    human_map: Dict[Tuple[str, int, str], object],
    baseline: str,
    target: str,
    outputs: Optional[Dict[int, Dict[str, str]]],
    scorer: CRScorer,
    inst_lookup: Dict[Tuple[str, int], ReviewInstance],
) -> dict:
    # Human paired deltas
    paired_before, paired_after = [], []
    for lang, idxs in overlap.items():
        for idx in idxs:
            base = human_map.get((lang, idx, baseline))
            tgt = human_map.get((lang, idx, target))
            if base and tgt and base.rel is not None and tgt.rel is not None:
                paired_before.append(base.rel)
                paired_after.append(tgt.rel)

    deltas = [a - b for a, b in zip(paired_after, paired_before)]

    # Correlation (CRScore vs human) for provided outputs (if any)
    corr_pairs: List[Tuple[float, float]] = []
    for lang, idxs in overlap.items():
        for idx in idxs:
            human_entry = human_map.get((lang, idx, target))
            if human_entry is None or human_entry.rel is None:
                continue
            review_text = ""
            if outputs and outputs.get(idx):
                review_text = outputs[idx]["review"]
            if not review_text:
                continue
            inst = inst_lookup.get((lang, idx))
            if not inst:
                continue
            cr_rel = scorer.score(inst.pseudo_refs, review_text).relevance
            corr_pairs.append((human_entry.rel, cr_rel))

    summary = {
        "overlap_counts": {lang: len(idxs) for lang, idxs in overlap.items()},
        "paired_count": len(paired_before),
        "paired_delta_mean": mean(deltas) if deltas else None,
        "correlation_count": len(corr_pairs),
        "correlation_pairs": corr_pairs[:5],  # sample to inspect
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase2 overlap analysis and optional CRScore correlation.")
    parser.add_argument("--raw-data", type=Path, required=True, help="Path to phase1 raw_data.json.")
    parser.add_argument("--phase2-dir", type=Path, required=True, help="Path to phase2 ratings directory.")
    parser.add_argument("--phase2-pattern", type=str, default="*review_qual*_final.csv", help="Glob for phase2 files.")
    parser.add_argument("--baseline-system", type=str, default="msg", help="Baseline system label in phase2 ratings.")
    parser.add_argument("--target-system", type=str, default="gpt3.5_pred", help="Target system label in phase2 ratings.")
    parser.add_argument("--outputs", type=Path, default=None, help="Optional system outputs JSONL (to correlate with human).")
    parser.add_argument("--tau", type=float, default=0.7314, help="CRScore tau for correlation scoring.")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="CRScore embedding model.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Where to write JSON summary.")
    args = parser.parse_args()

    overlap = find_overlap_ids(args.phase2_dir, args.phase2_pattern, args.baseline_system, args.target_system)
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    inst_lookup = {(inst.lang, inst.idx): inst for inst in instances}

    human_map = load_phase2_scores(sorted(args.phase2_dir.glob(args.phase2_pattern)))
    outputs_map = read_outputs(args.outputs) if args.outputs else None
    scorer = CRScorer(model_path=args.model_path, tau=args.tau)

    summary = summarize(overlap, human_map, args.baseline_system, args.target_system, outputs_map, scorer, inst_lookup)

    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))

    # Also emit overlap ids to help rerun systems on this subset
    overlap_out = args.summary_out.with_suffix(".ids.json") if args.summary_out else args.raw_data.parent / "phase2_overlap_ids.json"
    overlap_out.write_text(json.dumps(overlap, indent=2))
    print(f"Wrote overlap ids to {overlap_out}")


if __name__ == "__main__":
    main()
