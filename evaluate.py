"""
Evaluate reviews (baseline or system outputs) using CRScore.

Usage examples:
- Score baseline human reviews on the test split:
    python evaluate.py --baseline-only --split test
- Score a system output file:
    python evaluate.py --outputs results/loop.jsonl --split test
- Write a JSON summary:
    python evaluate.py --outputs results/loop.jsonl --summary-out results/loop_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language
from core.scoring import CRScorer
from core.loop import LoopConfig


def read_outputs(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing outputs file: {path}")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        inst = rec.get("instance", {})
        idx = int(inst.get("idx"))
        out[idx] = {"review": rec.get("best_review", ""), "lang": inst.get("lang", "")}
    return out


def evaluate_split(
    instances: List[ReviewInstance],
    scorer: CRScorer,
    outputs: Optional[Dict[int, Dict[str, str]]],
) -> Dict[str, Dict[str, float]]:
    per_lang: Dict[str, List[float]] = {}
    per_lang_con: Dict[str, List[float]] = {}
    per_lang_comp: Dict[str, List[float]] = {}

    for inst in instances:
        cand_review = inst.review
        if outputs and inst.idx in outputs:
            cand_review = outputs[inst.idx]["review"] or cand_review
        score = scorer.score(inst.pseudo_refs, cand_review)
        lang = inst.lang
        per_lang.setdefault(lang, []).append(score.relevance)
        per_lang_con.setdefault(lang, []).append(score.conciseness)
        per_lang_comp.setdefault(lang, []).append(score.comprehensiveness)

    summary: Dict[str, Dict[str, float]] = {}
    all_rels, all_cons, all_comps = [], [], []
    for lang, rels in per_lang.items():
        summary[lang] = {
            "Rel": mean(rels),
            "Con": mean(per_lang_con[lang]),
            "Comp": mean(per_lang_comp[lang]),
            "N": len(rels),
        }
        all_rels.extend(rels)
        all_cons.extend(per_lang_con[lang])
        all_comps.extend(per_lang_comp[lang])
    summary["overall"] = {
        "Rel": mean(all_rels) if all_rels else 0.0,
        "Con": mean(all_cons) if all_cons else 0.0,
        "Comp": mean(all_comps) if all_comps else 0.0,
        "N": len(all_rels),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reviews with CRScore.")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--outputs", type=Path, default=None, help="Path to system outputs JSONL (from run.py).")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="CRScore similarity threshold.")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="SentenceTransformer model path or HF id.")
    parser.add_argument("--baseline-only", action="store_true", help="Score human seed reviews (ignores --outputs).")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path to write summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]

    outputs_map = None
    if not args.baseline_only and args.outputs:
        outputs_map = read_outputs(args.outputs)
    elif not args.baseline_only and not args.outputs:
        raise ValueError("Provide --outputs or set --baseline-only.")

    scorer = CRScorer(model_path=args.model_path, tau=args.tau)
    summary = evaluate_split(selected, scorer, outputs_map)

    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
