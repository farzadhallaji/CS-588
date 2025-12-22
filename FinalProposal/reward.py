"""
Reward aggregation: SoftCRScore + evidence + regularizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .soft_crscore import SoftCRScoreResult
from .evidence_penalty import EvidencePenaltyResult


@dataclass
class RewardWeights:
    w_rel: float = 1.0
    w_unsupported: float = 0.6
    w_len: float = 0.02
    w_copy: float = 0.15
    len_norm: int = 400


@dataclass
class RewardBreakdown:
    reward: float
    soft_f1: float
    unsupported_rate: float
    len_penalty: float
    copy_ratio: float

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "soft_f1": self.soft_f1,
            "unsupported_rate": self.unsupported_rate,
            "len_penalty": self.len_penalty,
            "copy_ratio": self.copy_ratio,
        }


def _copy_ratio(review: str, pseudo_refs: Sequence[str]) -> float:
    if not pseudo_refs:
        return 0.0
    review_tokens = review.lower().split()
    ref_tokens = " ".join(pseudo_refs).lower().split()
    if not review_tokens or not ref_tokens:
        return 0.0
    inter = len(set(review_tokens) & set(ref_tokens))
    return inter / max(len(review_tokens), 1)


def _len_penalty(review: str, len_norm: int) -> float:
    return max(len(review) - len_norm, 0) / max(len_norm, 1)


def compute_reward(
    soft_scores: SoftCRScoreResult,
    evidence_res: EvidencePenaltyResult,
    review_text: str,
    pseudo_refs: Sequence[str],
    weights: RewardWeights,
) -> RewardBreakdown:
    len_pen = _len_penalty(review_text, weights.len_norm)
    copy = _copy_ratio(review_text, pseudo_refs)
    reward = (
        weights.w_rel * soft_scores.soft_f1
        - weights.w_unsupported * evidence_res.unsupported_rate
        - weights.w_len * len_pen
        - weights.w_copy * copy
    )
    return RewardBreakdown(
        reward=reward,
        soft_f1=soft_scores.soft_f1,
        unsupported_rate=evidence_res.unsupported_rate,
        len_penalty=len_pen,
        copy_ratio=copy,
    )
