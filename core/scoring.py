from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "sentence-transformers is required. Install via `pip install sentence-transformers torch`."
    ) from exc

from .utils import sentence_split


def compute_scores_from_sim(sim_matrix: np.ndarray, thresh: float) -> Tuple[float, float]:
    """
    Compute conciseness/precision-like (Con) and comprehensiveness/recall-like (Comp)
    given a similarity matrix of shape (claims x review_sentences).
    """
    if sim_matrix.size == 0:
        return 0.0, 0.0
    mask = sim_matrix > thresh
    prec_mask = (sim_matrix.max(axis=0) > thresh).astype(float)
    conciseness = float(prec_mask.mean()) if prec_mask.size else 0.0
    rec_mask = (mask.sum(axis=1) > 0).astype(float)
    comprehensiveness = float(rec_mask.mean()) if rec_mask.size else 0.0
    return conciseness, comprehensiveness


@dataclass
class ScoreResult:
    conciseness: float
    comprehensiveness: float
    relevance: float
    sim_matrix: np.ndarray
    review_sentences: List[str]

    def to_dict(self):
        return {"Con": self.conciseness, "Comp": self.comprehensiveness, "Rel": self.relevance}


class CRScorer:
    """
    Minimal CRScore-style scorer: SentenceTransformer embeddings + cosine sim.
    """

    def __init__(self, model_path: str = "mixedbread-ai/mxbai-embed-large-v1", tau: float = 0.6):
        self.tau = tau
        self.sbert = SentenceTransformer(model_path)

    def score(self, pseudo_refs: List[str], review: str) -> ScoreResult:
        review_sents = sentence_split(review) or [review]
        if not pseudo_refs:
            return ScoreResult(0.0, 0.0, 0.0, np.zeros((0, len(review_sents))), review_sents)
        claims = pseudo_refs
        claim_vecs = self.sbert.encode(claims, convert_to_tensor=True, show_progress_bar=False)
        review_vecs = self.sbert.encode(review_sents, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(claim_vecs, review_vecs).cpu().numpy()
        con, comp = compute_scores_from_sim(sims, self.tau)
        rel = 0.0 if (con + comp) == 0 else (2 * con * comp) / (con + comp)
        return ScoreResult(con, comp, rel, sims, review_sents)

    def max_sim(self, texts: Iterable[str], evidence: Iterable[str]) -> np.ndarray:
        a = list(texts)
        b = list(evidence)
        if not a or not b:
            return np.zeros((len(a), len(b)))
        a_vecs = self.sbert.encode(a, convert_to_tensor=True, show_progress_bar=False)
        b_vecs = self.sbert.encode(b, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(a_vecs, b_vecs).cpu().numpy()
        return sims
