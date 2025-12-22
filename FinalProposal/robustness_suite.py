"""
Robustness checks: claim dropout, claim noise, and evidence removal.
Outputs a CSV with deltas in Rel/Con/Comp and unsupported_rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple
import sys

import numpy as np

from . import PROPOSED_ROOT

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

from core.data import build_instances, load_raw_data  # type: ignore
from core.scoring import CRScorer  # type: ignore
from core.utils import sentence_split  # type: ignore
from core.evidence import EvidenceRetriever  # type: ignore

from .evidence_penalty import evidence_penalty, flatten_evidence_map
from .soft_crscore import embed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness stress tests.")
    parser.add_argument("--outputs", type=Path, required=True, help="System outputs JSONL (best_review field).")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--tau", type=float, default=0.7314)
    parser.add_argument("--drop-rates", type=str, default="0.2,0.4", help="Comma-separated claim dropout rates.")
    parser.add_argument("--fake-claims", type=str, default="ensure logging added for errors,performance optimized for large inputs")
    parser.add_argument("--fake-tau", type=float, default=0.35, help="Threshold to flag a fake claim as hallucinated.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).resolve().parent / "results" / "robustness.csv")
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


def cr_metrics(scorer: CRScorer, claims: List[str], review: str) -> Tuple[float, float, float]:
    score = scorer.score(claims, review)
    return score.relevance, score.conciseness, score.comprehensiveness


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    drop_rates = [float(x) for x in args.drop_rates.split(",") if x.strip()]
    fake_claims = [c.strip() for c in args.fake_claims.split(",") if c.strip()]

    records = load_raw_data(args.raw_data)
    inst_map = {inst.idx: inst for inst in build_instances(records)}
    outputs = load_outputs(args.outputs)
    scorer = CRScorer(model_path=args.model_path, tau=args.tau)
    retriever = EvidenceRetriever()

    summaries: List[Dict[str, object]] = []

    for idx, rec in outputs.items():
        inst = inst_map.get(idx)
        if inst is None:
            continue
        review = rec.get("best_review") or inst.review
        base_claims = inst.pseudo_refs
        base_rel, base_con, base_comp = cr_metrics(scorer, base_claims, review)
        evidence_map = retriever.retrieve(base_claims, inst.patch, inst.old_file)
        evidence_lines = flatten_evidence_map(evidence_map)
        sent_split = sentence_split(review)
        base_ev = evidence_penalty(scorer, sent_split, evidence_lines, margin=args.tau * 0.5)
        base_fake_hit = 0.0
        if fake_claims:
            sims = scorer.max_sim(fake_claims, sent_split)
            base_fake_hit = float(np.mean((sims.max(axis=1) > args.fake_tau).astype(float))) if sims.size else 0.0

        # Base row
        summaries.append(
            {
                "idx": idx,
                "lang": inst.lang,
                "scenario": "base",
                "rel_delta": 0.0,
                "con_delta": 0.0,
                "comp_delta": 0.0,
                "unsupported_rate": base_ev.unsupported_rate,
                "fake_hit_rate": base_fake_hit,
            }
        )

        for rate in drop_rates:
            keep_n = max(1, int((1 - rate) * len(base_claims))) if base_claims else 0
            dropped_claims = rng.sample(base_claims, keep_n) if base_claims else []
            rel, con, comp = cr_metrics(scorer, dropped_claims, review)
            summaries.append(
                {
                    "idx": idx,
                    "lang": inst.lang,
                    "scenario": f"drop_{int(rate*100)}",
                    "rel_delta": rel - base_rel,
                    "con_delta": con - base_con,
                    "comp_delta": comp - base_comp,
                    "unsupported_rate": base_ev.unsupported_rate,
                    "fake_hit_rate": base_fake_hit,
                }
            )

        if fake_claims:
            noisy_claims = base_claims + fake_claims
            rel, con, comp = cr_metrics(scorer, noisy_claims, review)
            sims = scorer.max_sim(fake_claims, sent_split)
            hit_rate = float(np.mean((sims.max(axis=1) > args.fake_tau).astype(float))) if sims.size else 0.0
            summaries.append(
                {
                    "idx": idx,
                    "lang": inst.lang,
                    "scenario": "fake_claims",
                    "rel_delta": rel - base_rel,
                    "con_delta": con - base_con,
                    "comp_delta": comp - base_comp,
                    "unsupported_rate": base_ev.unsupported_rate,
                    "fake_hit_rate": hit_rate,
                }
            )

        # Evidence removal: recompute unsupported with empty evidence
        ev_removed = evidence_penalty(scorer, sent_split, [], margin=args.tau * 0.5)
        summaries.append(
            {
                "idx": idx,
                "lang": inst.lang,
                "scenario": "no_evidence",
                "rel_delta": 0.0,
                "con_delta": 0.0,
                "comp_delta": 0.0,
                "unsupported_rate": ev_removed.unsupported_rate,
                "fake_hit_rate": base_fake_hit,
            }
        )

    # Aggregate per scenario
    agg_rows: List[Dict[str, object]] = []
    by_scenario: Dict[str, List[Dict[str, object]]] = {}
    for row in summaries:
        by_scenario.setdefault(row["scenario"], []).append(row)
    for scen, rows in by_scenario.items():
        agg_rows.append(
            {
                "scenario": scen,
                "count": len(rows),
                "rel_delta_mean": mean([r["rel_delta"] for r in rows]) if rows else 0.0,
                "con_delta_mean": mean([r["con_delta"] for r in rows]) if rows else 0.0,
                "comp_delta_mean": mean([r["comp_delta"] for r in rows]) if rows else 0.0,
                "unsupported_rate_mean": mean([r["unsupported_rate"] for r in rows]) if rows else 0.0,
                "fake_hit_rate_mean": mean([r["fake_hit_rate"] for r in rows]) if rows else 0.0,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        fieldnames = [
            "scenario",
            "count",
            "rel_delta_mean",
            "con_delta_mean",
            "comp_delta_mean",
            "unsupported_rate_mean",
            "fake_hit_rate_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    print(f"Wrote robustness summary to {args.output_csv}")


if __name__ == "__main__":
    main()
