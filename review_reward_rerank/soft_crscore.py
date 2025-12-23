"""
SoftCRScore: differentiable-ish variant of CRScore over sentence embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from scipy.special import expit  # stable sigmoid

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise RuntimeError(
        "sentence-transformers is required for review_reward_rerank. Install via `pip install sentence-transformers torch`."
    ) from exc

EPS = 1e-8


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + EPS
    return mat / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    return np.matmul(a_norm, b_norm.T)


@dataclass
class SoftCRScoreResult:
    soft_precision: float
    soft_recall: float
    soft_f1: float
    alignments: List[List[dict]]
    sim_matrix: np.ndarray

    def to_dict(self) -> dict:
        return {
            "soft_precision": self.soft_precision,
            "soft_recall": self.soft_recall,
            "soft_f1": self.soft_f1,
            "soft_con_precision": self.soft_precision,
            "soft_comp_recall": self.soft_recall,
            "soft_rel_f1": self.soft_f1,
            "alignments": self.alignments,
        }


def embed_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    vecs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs)


def soft_crscore(
    sent_embs: np.ndarray,
    claim_embs: np.ndarray,
    tau: float = 0.7314,
    temp: float = 0.05,
    top_k: int = 2,
) -> SoftCRScoreResult:
    """
    Soft matching between review sentences and claims.
    """
    if sent_embs.size == 0 or claim_embs.size == 0:
        return SoftCRScoreResult(0.0, 0.0, 0.0, [], np.zeros((sent_embs.shape[0], claim_embs.shape[0])))

    sim = cosine_matrix(sent_embs, claim_embs)
    act = expit((sim - tau) / max(temp, EPS))

    soft_precision = float(act.max(axis=1).mean()) if act.size else 0.0
    soft_recall = float(act.max(axis=0).mean()) if act.size else 0.0
    denom = soft_precision + soft_recall + EPS
    soft_f1 = 0.0 if denom == 0 else float(2 * soft_precision * soft_recall / denom)

    alignments: List[List[dict]] = []
    for i in range(act.shape[0]):
        row = act[i]
        if row.size == 0:
            alignments.append([])
            continue
        top_idx = np.argsort(-row)[: max(top_k, 1)]
        alignments.append([{"claim_idx": int(j), "score": float(row[j])} for j in top_idx])

    return SoftCRScoreResult(soft_precision, soft_recall, soft_f1, alignments, sim)
