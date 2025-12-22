"""
Build preference pairs automatically from scored candidates using the reward function.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

from . import PROPOSED_ROOT

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

from core.scoring import CRScorer  # type: ignore
from core.utils import sentence_split  # type: ignore

from .soft_crscore import embed_texts, soft_crscore
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
    parser.add_argument("--w-rel", type=float, default=1.0)
    parser.add_argument("--w-unsupported", type=float, default=0.6)
    parser.add_argument("--w-len", type=float, default=0.02)
    parser.add_argument("--w-copy", type=float, default=0.15)
    parser.add_argument("--len-norm", type=int, default=400)
    return parser.parse_args()


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
        sent_embs = embed_texts(scorer.sbert, sentences)
        soft_res = soft_crscore(sent_embs, claim_embs, tau=args.tau, temp=args.temp)
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

            top, bottom = ranked[0], ranked[-1]
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

            write_pair(top, bottom, "top_vs_bottom")
            if args.include_median and median is not top and median is not bottom:
                write_pair(top, median, "top_vs_median")


if __name__ == "__main__":
    main()
