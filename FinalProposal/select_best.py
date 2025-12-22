"""
Select the best candidate review using SoftCRScore + evidence penalty.
Outputs evaluation-compatible JSONL (instance + best_review).
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
from .reward import RewardBreakdown, RewardWeights, compute_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best review candidate with SoftCRScore + evidence penalty.")
    parser.add_argument("--candidates", type=Path, required=True, help="Path to candidates JSONL from generate_candidates.py")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="SentenceTransformer model.")
    parser.add_argument("--tau", type=float, default=0.7314, help="CRScore threshold; keep consistent with evaluate.py/human_eval.py.")
    parser.add_argument("--temp", type=float, default=0.05, help="SoftCRScore temperature.")
    parser.add_argument("--evidence-margin", type=float, default=0.35, help="Similarity margin below which a sentence is unsupported.")
    parser.add_argument("--top-k-align", type=int, default=2, help="Top-k claim alignments to keep per sentence.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on instances for quick runs.")
    parser.add_argument("--w-rel", type=float, default=1.0)
    parser.add_argument("--w-unsupported", type=float, default=0.6)
    parser.add_argument("--w-len", type=float, default=0.02)
    parser.add_argument("--w-copy", type=float, default=0.15)
    parser.add_argument("--len-norm", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "selected.jsonl")
    return parser.parse_args()


def score_candidate(
    scorer: CRScorer,
    text: str,
    claims: List[str],
    claim_embs,
    evidence_lines: List[str],
    args: argparse.Namespace,
    weights: RewardWeights,
) -> Dict[str, object]:
    sentences = sentence_split(text)
    sent_embs = embed_texts(scorer.sbert, sentences)
    soft_res = soft_crscore(sent_embs, claim_embs, tau=args.tau, temp=args.temp, top_k=args.top_k_align)
    evidence_res = evidence_penalty(scorer, sentences, evidence_lines, margin=args.evidence_margin)
    reward_bd = compute_reward(soft_res, evidence_res, text, claims, weights)
    alignments = []
    for i, sent in enumerate(sentences):
        aligns = soft_res.alignments[i] if i < len(soft_res.alignments) else []
        alignments.append({"sentence": sent, "claims": aligns})
    return {
        "text": text,
        "reward": reward_bd.to_dict(),
        "soft_scores": soft_res.to_dict(),
        "evidence": evidence_res.to_dict(),
        "alignments": alignments,
    }


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
                best_review = seed
                selection = {
                    "reward": None,
                    "soft_scores": None,
                    "unsupported_rate": None,
                    "chosen_from": 0,
                }
                output = {
                    "instance": rec.get("instance", {}),
                    "seed_review": seed,
                    "best_review": best_review,
                    "selection": selection,
                    "alignments": [],
                    "evidence_map": evidence_map,
                }
                fout.write(json.dumps(output) + "\n")
                fout.flush()
                continue

            scored = [
                score_candidate(
                    scorer=scorer,
                    text=cand.get("text", "").strip() or seed,
                    claims=claims,
                    claim_embs=claim_embs,
                    evidence_lines=evidence_lines,
                    args=args,
                    weights=weights,
                )
                for cand in candidates
            ]
            best = max(scored, key=lambda x: x["reward"]["reward"])
            fout.write(
                json.dumps(
                    {
                        "instance": rec.get("instance", {}),
                        "seed_review": seed,
                        "best_review": best["text"],
                        "selection": {
                            "reward": best["reward"],
                            "soft_scores": best["soft_scores"],
                            "unsupported_rate": best["evidence"]["unsupported_rate"],
                            "chosen_from": len(scored),
                        },
                        "alignments": best["alignments"],
                        "evidence_map": best["evidence"].get("evidence_map", {}),
                        "candidates_scored": scored,
                    }
                )
                + "\n"
            )
            fout.flush()


if __name__ == "__main__":
    main()
