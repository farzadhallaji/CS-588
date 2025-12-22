"""
Build preference pairs automatically from scored candidates using the reward function.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import numpy as np

from . import PROPOSED_ROOT

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

from core.scoring import CRScorer, ScoreResult  # type: ignore
from core.utils import sentence_split  # type: ignore

from .soft_crscore import SoftCRScoreResult, embed_texts, soft_crscore
from .evidence_penalty import collect_evidence, flatten_evidence_map, evidence_penalty
from .reward import RewardWeights, compute_reward
from .prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create preference pairs from candidates.")
    parser.add_argument("--candidates", type=Path, required=True, help="Path to candidates JSONL.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "preferences.jsonl")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--tau", type=float, default=0.7314)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--evidence-margin", type=float, default=0.35)
    parser.add_argument("--include-median", action="store_true", help="Also add top vs median preference pairs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-mode", choices=["soft", "hard"], default="soft", help="soft: SoftCRScore (default); hard: CRScore precision/recall.")
    parser.add_argument("--top-k-align", type=int, default=2, help="Top-k claim alignments to keep per sentence.")
    parser.add_argument("--top-k", type=int, default=1, help="How many top candidates to include in pairs.")
    parser.add_argument("--bottom-k", type=int, default=1, help="How many bottom candidates to include in pairs.")
    parser.add_argument("--w-rel", type=float, default=1.0)
    parser.add_argument("--w-unsupported", type=float, default=0.6)
    parser.add_argument("--w-len", type=float, default=0.02)
    parser.add_argument("--w-copy", type=float, default=0.15)
    parser.add_argument("--len-norm", type=int, default=400)
    return parser.parse_args()


def hard_to_soft(score: ScoreResult, top_k: int) -> SoftCRScoreResult:
    sim = score.sim_matrix.T  # sentences x claims
    alignments: List[List[dict]] = []
    for i in range(sim.shape[0]):
        row = sim[i]
        if row.size == 0:
            alignments.append([])
            continue
        top_idx = np.argsort(-row)[: max(top_k, 1)]
        alignments.append([{"claim_idx": int(j), "score": float(row[j])} for j in top_idx])
    return SoftCRScoreResult(
        soft_precision=score.conciseness,
        soft_recall=score.comprehensiveness,
        soft_f1=score.relevance,
        alignments=alignments,
        sim_matrix=sim,
    )


def score_candidates(
    scorer: CRScorer,
    claims: List[str],
    claim_embs,
    evidence_lines: List[str],
    candidates: List[dict],
    seed_review: str,
    args: argparse.Namespace,
    weights: RewardWeights,
) -> List[dict]:
    scored = []
    for cand in candidates:
        text = cand.get("text", "").strip() or seed_review
        sentences = sentence_split(text)
        if args.score_mode == "hard":
            hard = scorer.score(claims, text)
            soft_res = hard_to_soft(hard, args.top_k_align)
        else:
            sent_embs = embed_texts(scorer.sbert, sentences)
            soft_res = soft_crscore(sent_embs, claim_embs, tau=args.tau, temp=args.temp, top_k=args.top_k_align)
        evidence_res = evidence_penalty(scorer, sentences, evidence_lines, margin=args.evidence_margin)
        reward_bd = compute_reward(soft_res, evidence_res, text, claims, weights)
        scored.append(
            {
                "text": text,
                "prompt_variant": cand.get("prompt_variant", ""),
                "reward": reward_bd.reward,
                "reward_breakdown": reward_bd.to_dict(),
                "soft_scores": soft_res.to_dict(),
                "evidence": evidence_res.to_dict(),
            }
        )
    return scored


def main() -> None:
    args = parse_args()
    weights = RewardWeights(
        w_rel=args.w_rel,
        w_unsupported=args.w_unsupported,
        w_len=args.w_len,
        w_copy=args.w_copy,
        len_norm=args.len_norm,
    )
    scorer = CRScorer(model_path=args.model_path, tau=args.tau)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.candidates.open() as fin, args.output.open("w") as fout:
        for i, line in enumerate(fin):
            if args.limit and i >= args.limit:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            claims = rec.get("claims", []) or []
            claim_embs = embed_texts(scorer.sbert, claims)
            patch = rec.get("patch", "")
            old_file = rec.get("old_file", "")
            evidence_map = rec.get("evidence") or collect_evidence(claims, patch, old_file)
            evidence_lines = flatten_evidence_map(evidence_map)
            candidates = rec.get("candidates", []) or []
            seed = rec.get("seed_review", "")
            if not candidates:
                continue

            scored = score_candidates(scorer, claims, claim_embs, evidence_lines, candidates, seed, args, weights)
            ranked = sorted(scored, key=lambda x: x["reward"], reverse=True)
            if not ranked:
                continue

            top = ranked[: max(1, args.top_k)]
            bottom = list(reversed(ranked[-max(1, args.bottom_k) :]))
            median = ranked[len(ranked) // 2]

            prompt_text = build_prompt(
                variant="default",
                seed_review=seed,
                claims=claims,
                diff=patch,
                old_code=old_file,
                uncovered_claims=[],
                offending_sentences=[],
                evidence_snippets=evidence_lines,
            )

            def write_pair(chosen: dict, rejected: dict, pair_type: str) -> None:
                fout.write(
                    json.dumps(
                        {
                            "prompt": prompt_text,
                            "chosen": chosen["text"],
                            "rejected": rejected["text"],
                            "meta": {
                                "idx": rec.get("instance", {}).get("idx"),
                                "lang": rec.get("instance", {}).get("lang"),
                                "pair_type": pair_type,
                                "prompt_variant_chosen": chosen.get("prompt_variant"),
                                "prompt_variant_rejected": rejected.get("prompt_variant"),
                                "reward_chosen": chosen.get("reward"),
                                "reward_rejected": rejected.get("reward"),
                            },
                        }
                    )
                    + "\n"
                )

            pair_label = f"top{max(1, args.top_k)}_vs_bottom{max(1, args.bottom_k)}"
            for i in range(min(len(top), len(bottom))):
                write_pair(top[i], bottom[i], pair_label)
            if args.include_median and median not in top and median not in bottom:
                write_pair(top[0], median, "top_vs_median")


if __name__ == "__main__":
    main()
