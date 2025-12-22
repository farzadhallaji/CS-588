"""
Evidence grounding penalty utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import sys

import numpy as np

from . import PROPOSED_ROOT

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

from core.evidence import EvidenceRetriever  # type: ignore
from core.scoring import CRScorer  # type: ignore

EPS = 1e-8


@dataclass
class EvidencePenaltyResult:
    unsupported_rate: float
    per_sentence_support: List[float]
    evidence_map: Dict[int, Dict[str, object]]

    def to_dict(self) -> dict:
        return {
            "unsupported_rate": self.unsupported_rate,
            "per_sentence_support": self.per_sentence_support,
            "evidence_map": self.evidence_map,
        }


def collect_evidence(pseudo_refs: Sequence[str], patch: str, old_file: str, max_lines_per_ref: int = 3) -> Dict[str, List[str]]:
    retriever = EvidenceRetriever(max_lines_per_ref=max_lines_per_ref)
    return retriever.retrieve(list(pseudo_refs), patch or "", old_file or "")


def flatten_evidence_map(evidence: Dict[str, List[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for _, lines in evidence.items():
        for ln in lines:
            if ln not in seen and ln.strip():
                seen.add(ln)
                ordered.append(ln.strip())
    return ordered


def evidence_penalty(
    scorer: CRScorer,
    review_sentences: Sequence[str],
    evidence_lines: Sequence[str],
    margin: float = 0.35,
) -> EvidencePenaltyResult:
    """
    Compute how well each sentence is supported by diff/old-code evidence.
    """
    if not review_sentences:
        return EvidencePenaltyResult(0.0, [], {})

    sims = scorer.max_sim(review_sentences, evidence_lines)
    if sims.size == 0:
        per_sent_support = [0.0 for _ in review_sentences]
    else:
        per_sent_support = [float(sims[i].max()) for i in range(sims.shape[0])]
    unsupported_rate = float(np.mean([1.0 if s < margin else 0.0 for s in per_sent_support])) if per_sent_support else 0.0

    evidence_map: Dict[int, Dict[str, object]] = {}
    for i, sent in enumerate(review_sentences):
        if not evidence_lines:
            evidence_map[i] = {"sentence": sent, "evidence": [], "max_sim": per_sent_support[i] if per_sent_support else 0.0}
            continue
        best_idx = int(np.argmax(sims[i])) if sims.size else -1
        best_ev = evidence_lines[best_idx] if evidence_lines and 0 <= best_idx < len(evidence_lines) else ""
        evidence_map[i] = {"sentence": sent, "evidence": best_ev, "max_sim": per_sent_support[i]}

    return EvidencePenaltyResult(unsupported_rate, per_sent_support, evidence_map)
