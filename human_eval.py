"""
Evaluate system outputs against phase2 human ratings and compute correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple
import sys

try:
    from scipy.stats import kendalltau, spearmanr, wilcoxon
except ModuleNotFoundError as exc:
    raise RuntimeError("scipy is required for human_eval.py. Install via `pip install scipy`.") from exc

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language
from core.scoring import CRScorer


@dataclass
class HumanAggregate:
    con: Optional[float]
    comp: Optional[float]
    rel: Optional[float]
    review: str
    count: int


def parse_lang_from_name(path: Path) -> str:
    stem = path.name.split("_")[0]
    if stem not in {"py", "java", "js"}:
        raise ValueError(f"Could not infer language from {path.name}")
    return stem


def load_phase2_scores(files: Iterable[Path]) -> Dict[Tuple[str, int, str], HumanAggregate]:
    buckets: Dict[Tuple[str, int, str], Dict[str, List[float]]] = {}
    reviews: Dict[Tuple[str, int, str], List[str]] = {}
    for path in files:
        if not path.exists():
            continue
        lang = parse_lang_from_name(path)
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                system = (row.get("system") or "").strip()
                if not system:
                    continue
                try:
                    idx = int(float(row.get("index", "")))
                except (TypeError, ValueError):
                    continue
                key = (lang, idx, system)
                buckets.setdefault(key, {"con": [], "comp": [], "rel": []})
                reviews.setdefault(key, [])
                for col, dest in [("Con (P)", "con"), ("Comp (R)", "comp"), ("Rel (F)", "rel")]:
                    val = row.get(col)
                    if val is None or str(val).strip() == "":
                        continue
                    try:
                        buckets[key][dest].append(float(val))
                    except ValueError:
                        continue
                text = (row.get("review") or "").strip()
                if text:
                    reviews[key].append(text)

    agg: Dict[Tuple[str, int, str], HumanAggregate] = {}
    for key, vals in buckets.items():
        agg[key] = HumanAggregate(
            con=mean(vals["con"]) if vals["con"] else None,
            comp=mean(vals["comp"]) if vals["comp"] else None,
            rel=mean(vals["rel"]) if vals["rel"] else None,
            review=reviews.get(key, [""])[0] if reviews.get(key) else "",
            count=max(len(vals["con"]), len(vals["comp"]), len(vals["rel"]), len(reviews.get(key, []))),
        )
    return agg


def read_outputs(path: Optional[Path]) -> Dict[int, Dict[str, str]]:
    if path is None:
        return {}
    outputs: Dict[int, Dict[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        inst = rec.get("instance", {}) or {}
        meta = inst.get("meta") or {}
        idx_val = meta.get("index", inst.get("idx"))
        try:
            idx = int(idx_val)
        except (TypeError, ValueError):
            continue
        outputs[idx] = {"review": rec.get("best_review", ""), "lang": inst.get("lang", "")}
    return outputs


def wilcoxon_effect_size(statistic: float, n: int) -> float:
    if n <= 0:
        return 0.0
    mu = n * (n + 1) / 4
    sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return 0.0
    z = (statistic - mu) / sigma
    return z / sqrt(n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 human evaluation stats.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json",
        help="Path to phase1 raw_data.json (pseudo-refs and seed reviews).",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=ROOT.parent / "CRScore" / "human_study" / "phase2",
        help="Directory containing phase2 human rating CSVs.",
    )
    parser.add_argument(
        "--phase2-pattern",
        type=str,
        default="*review_qual*_final.csv",
        help="Glob to select phase2 files.",
    )
    parser.add_argument("--outputs", type=Path, default=None, help="System output JSONL to align with human ratings.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--tau", type=float, default=0.6, help="CRScore similarity threshold.")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="SentenceTransformer model.")
    parser.add_argument(
        "--baseline-system",
        type=str,
        default="msg",
        help="System label in phase2 CSVs for the seed review (e.g., msg).",
    )
    parser.add_argument(
        "--target-system",
        type=str,
        default="gpt3.5_pred",
        help="System label in phase2 CSVs to compare against the baseline.",
    )
    parser.add_argument(
        "--corr-system",
        type=str,
        default=None,
        help="System label to use for correlation (defaults to --target-system).",
    )
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional path to write JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corr_system = args.corr_system or args.target_system

    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    inst_lookup: Dict[Tuple[str, int], ReviewInstance] = {(inst.lang, inst.idx): inst for inst in selected}

    phase2_files = sorted(args.phase2_dir.glob(args.phase2_pattern))
    human_map = load_phase2_scores(phase2_files)
    outputs_map = read_outputs(args.outputs)
    scorer = CRScorer(model_path=args.model_path, tau=args.tau)

    paired_before, paired_after = [], []
    for (lang, idx), inst in inst_lookup.items():
        base = human_map.get((lang, idx, args.baseline_system))
        tgt = human_map.get((lang, idx, args.target_system))
        if base and tgt and base.rel is not None and tgt.rel is not None:
            paired_before.append(base.rel)
            paired_after.append(tgt.rel)

    deltas = [a - b for a, b in zip(paired_after, paired_before)]
    w_stat, w_p, effect = None, None, None
    n_eff = len([d for d in deltas if d != 0])
    if paired_before and paired_after:
        w_stat, w_p = wilcoxon(paired_after, paired_before, zero_method="wilcox")
        effect = wilcoxon_effect_size(w_stat, n_eff)

    corr_pairs: List[Tuple[float, float]] = []
    for (lang, idx), inst in inst_lookup.items():
        human_entry = human_map.get((lang, idx, corr_system))
        if human_entry is None or human_entry.rel is None:
            continue
        review_text = ""
        if outputs_map.get(idx):
            review_text = outputs_map[idx]["review"]
        if not review_text:
            review_text = human_entry.review or inst.review
        cr_rel = scorer.score(inst.pseudo_refs, review_text).relevance
        corr_pairs.append((human_entry.rel, cr_rel))

    spearman_res = spearmanr([h for h, _ in corr_pairs], [c for _, c in corr_pairs]) if corr_pairs else None
    kendall_res = kendalltau([h for h, _ in corr_pairs], [c for _, c in corr_pairs]) if corr_pairs else None

    summary = {
        "split": args.split,
        "baseline_system": args.baseline_system,
        "target_system": args.target_system,
        "correlation_system": corr_system,
        "paired_count": len(paired_before),
        "paired_delta_mean": mean(deltas) if deltas else None,
        "wilcoxon": {"statistic": w_stat, "pvalue": w_p, "effect_size_r": effect, "n_effective": n_eff},
        "correlation": {
            "count": len(corr_pairs),
            "spearman": {"corr": getattr(spearman_res, "correlation", None), "pvalue": getattr(spearman_res, "pvalue", None)}
            if spearman_res
            else None,
            "kendall": {"corr": getattr(kendall_res, "correlation", None), "pvalue": getattr(kendall_res, "pvalue", None)}
            if kendall_res
            else None,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
