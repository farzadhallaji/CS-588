from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

from .data import ReviewInstance
from .evidence import EvidenceRetriever
from .scoring import CRScorer, ScoreResult
from .utils import lexical_overlap, sentence_change_ratio, sentence_split
from .editors import BaseEditor

BASE_SYSTEM_PROMPT = (
    "You are a senior code reviewer. Improve the given review comment.\n"
    "Constraints:\n"
    "* Be specific to the diff/claims provided.\n"
    "* Do NOT invent facts, files, or behaviors not supported by the inputs.\n"
    "* Prefer short, high-signal feedback (correctness, edge cases, tests, risks).\n"
    "* Avoid generic advice and style nits unless they directly affect correctness.\n"
    "Output ONLY the revised review text (no explanations, no bullets unless the review itself uses bullets)."
)

LOOP_INSTRUCTIONS = {
    "loop": (
        "Task: minimally edit the current review.\n"
        "Goals (in priority order):\n"
        "1. Cover the uncovered pseudo-references using the evidence snippets when available.\n"
        "2. Remove or rewrite the offending low-similarity sentences.\n"
        "3. Keep or improve precision: do not add unrelated points.\n"
        "Rules:\n"
        "* If evidence does not support a new statement, do not add it.\n"
        "* If you are unsure, suggest a test instead of asserting behavior.\n"
        "Output only the revised review."
    ),
    "k1": (
        "Task: produce your best revision in ONE pass.\n"
        "Include the most important uncovered items and delete low-signal/off-topic content.\n"
        "Do not add anything unsupported by evidence or the diff.\n"
        "Output only the revised review."
    ),
    "rewrite": (
        "Task: rewrite the review from scratch for maximum quality.\n"
        "Include the key uncovered pseudo-references and the most important risks/tests.\n"
        "Do not invent facts; ground statements in the diff/evidence.\n"
        "Keep it concise (prefer 1-4 sentences).\n"
        "Output only the revised review."
    ),
    "no-evidence": (
        "Task: improve the review using ONLY the current review + uncovered pseudo-references + the diff context implied by them.\n"
        "Do NOT cite or rely on evidence (assume it is unavailable).\n"
        "Rules:\n"
        "* Do not assert specific code behavior unless it is explicitly stated in the pseudo-references.\n"
        "* Prefer concrete test suggestions over factual claims.\n"
        "Keep it short and specific. Output only the revised review."
    ),
    "no-selection": (
        "Generate an alternative valid revision that is still faithful to the inputs, but phrased differently.\n"
        "Keep the same constraints: no hallucinations, concise, specific, test-focused."
    ),
}


@dataclass
class LoopConfig:
    max_iter: int = 3
    num_samples: int = 2
    tau: float = 0.7314
    tau_evidence: float = 0.35
    precision_drop: float = 0.05
    max_sentence_change: float = 0.7
    length_budget: int = 900
    lambda_len: float = 5e-4
    lambda_copy: float = 0.05
    epsilon: float = 0.002
    rewrite: bool = False
    disable_evidence: bool = False
    selection: str = "crscore"  # crscore|random|shortest
    prompt_style: str = "loop"  # loop|k1|rewrite|no-evidence|no-selection
    system_prompt: str = BASE_SYSTEM_PROMPT
    prompt_text: str = ""


def build_prompt(
    current_review: str,
    uncovered: List[str],
    offending: List[str],
    evidence: Dict[str, List[str]],
    length_budget: int,
    rewrite: bool,
    instructions: str,
) -> str:
    parts = [
        "Current review:",
        current_review.strip(),
        "",
        "Uncovered pseudo-references:",
    ]
    if uncovered:
        parts.extend([f"- {u}" for u in uncovered])
    else:
        parts.append("- None (focus on conciseness and precision).")
    if offending:
        parts.append("")
        parts.append("Offending (low-similarity) sentences to trim or rewrite:")
        parts.extend([f"- {s}" for s in offending])
    parts.append("")
    parts.append("Evidence snippets:")
    if evidence:
        for ref, lines in evidence.items():
            ev_str = " | ".join(lines) if lines else "(none)"
            parts.append(f"- {ref}: {ev_str}")
    else:
        parts.append("- (no evidence available)")
    parts.append("")
    instr_prefix = instructions or ("Rewrite freely for quality." if rewrite else "Edit minimally.")
    parts.append(f"{instr_prefix}\nKeep within {length_budget} characters. Do not invent unsupported claims. Output only the revised review.")
    return "\n".join(parts)


class IterativeRefiner:
    def __init__(self, scorer: CRScorer, editor: BaseEditor, config: LoopConfig):
        self.scorer = scorer
        self.editor = editor
        self.config = config

    def _evidence_guardrail(
        self, new_sentences: List[str], evidence_lines: List[str], tau: float
    ) -> bool:
        # If nothing new is added, allow it.
        if not new_sentences:
            return True
        # If we have no evidence, do not allow unsupported additions.
        if not evidence_lines:
            return False
        sims = self.scorer.max_sim(new_sentences, evidence_lines)
        if sims.size == 0:
            return False
        for i, sent in enumerate(new_sentences):
            # substring support
            if any(ev and ev.lower() in sent.lower() for ev in evidence_lines):
                continue
            # semantic support
            if sims[i].max() < tau:
                return False
        return True

    def _copy_penalty(self, review: str, pseudo_refs: List[str]) -> float:
        if not pseudo_refs:
            return 0.0
        review_tokens = review.lower().split()
        ref_tokens = " ".join(pseudo_refs).lower().split()
        if not review_tokens or not ref_tokens:
            return 0.0
        inter = len(set(review_tokens) & set(ref_tokens))
        return inter / max(len(review_tokens), 1)

    def run(self, instance: ReviewInstance) -> Dict[str, object]:
        cfg = self.config
        current = instance.review.strip()
        best_review = current
        best_score = self.scorer.score(instance.pseudo_refs, current)
        history: List[Dict[str, object]] = []
        no_improve_steps = 0
        evidence_map = {}

        for t in range(cfg.max_iter):
            start = time.time()
            prev_best_rel = best_score.relevance
            score = self.scorer.score(instance.pseudo_refs, current)
            covered = score.sim_matrix.max(axis=1) if score.sim_matrix.size else np.array([])
            uncovered = [p for i, p in enumerate(instance.pseudo_refs) if covered.size == 0 or covered[i] < cfg.tau]
            sent_scores = score.sim_matrix.max(axis=0) if score.sim_matrix.size else np.array([])
            offending = [s for i, s in enumerate(score.review_sentences) if sent_scores.size == 0 or sent_scores[i] < cfg.tau]

            if not evidence_map:
                evidence_map = EvidenceRetriever().retrieve(instance.pseudo_refs, instance.patch, instance.old_file)
            prompt = build_prompt(
                current_review=current,
                uncovered=uncovered,
                offending=offending,
                evidence=evidence_map,
                length_budget=cfg.length_budget,
                rewrite=cfg.rewrite,
                instructions=cfg.prompt_text,
            )
            candidates = self.editor.propose(
                current_review=current,
                uncovered=uncovered,
                offending=offending,
                evidence=evidence_map,
                prompt=prompt,
                num_samples=cfg.num_samples,
            )

            evaluated_candidates = []
            for cand in candidates:
                cand_text = cand.strip()
                cand_score = self.scorer.score(instance.pseudo_refs, cand_text)
                change_ratio = 0.0 if cfg.rewrite else sentence_change_ratio(score.review_sentences, sentence_split(cand_text))
                con_ok = cand_score.conciseness + 1e-6 >= max(0.0, score.conciseness - cfg.precision_drop)
                len_ok = len(cand_text) <= cfg.length_budget
                evidence_lines = [ln for lst in evidence_map.values() for ln in lst]
                new_sentences = [s for s in sentence_split(cand_text) if s not in score.review_sentences]
                evidence_ok = cfg.disable_evidence or self._evidence_guardrail(new_sentences, evidence_lines, cfg.tau_evidence)
                min_edit_ok = cfg.rewrite or change_ratio <= cfg.max_sentence_change
                valid = con_ok and len_ok and evidence_ok and min_edit_ok
                objective = cand_score.relevance - cfg.lambda_len * len(cand_text) - cfg.lambda_copy * self._copy_penalty(
                    cand_text, instance.pseudo_refs
                )
                evaluated_candidates.append(
                    {
                        "text": cand_text,
                        "score": cand_score.to_dict(),
                        "change_ratio": change_ratio,
                        "con_ok": con_ok,
                        "len_ok": len_ok,
                        "evidence_ok": evidence_ok,
                        "min_edit_ok": min_edit_ok,
                        "valid": valid,
                        "objective": objective,
                    }
                )

            valid_candidates = [c for c in evaluated_candidates if c["valid"]]
            if not valid_candidates:
                history.append(
                    {
                        "iter": t,
                        "score": score.to_dict(),
                        "uncovered": uncovered,
                        "offending": offending,
                        "candidates": evaluated_candidates,
                        "selected": None,
                        "elapsed_sec": time.time() - start,
                    }
                )
                break

            if cfg.selection == "random":
                selected = random.choice(valid_candidates)
            elif cfg.selection == "shortest":
                selected = min(valid_candidates, key=lambda c: len(c["text"]))
            else:
                selected = max(valid_candidates, key=lambda c: c["objective"])

            current = selected["text"]
            history.append(
                {
                    "iter": t,
                    "score": score.to_dict(),
                    "uncovered": uncovered,
                    "offending": offending,
                    "candidates": evaluated_candidates,
                    "selected": selected,
                    "elapsed_sec": time.time() - start,
                }
            )

            if selected["score"]["Rel"] > best_score.relevance:
                best_score = self.scorer.score(instance.pseudo_refs, current)
                best_review = current
            improved = best_score.relevance - prev_best_rel
            if improved < cfg.epsilon:
                no_improve_steps += 1
            else:
                no_improve_steps = 0
            if no_improve_steps >= 2:
                break

        return {
            "instance": {
                "idx": instance.idx,
                "lang": instance.lang,
                "meta": instance.meta,
            },
            "config": asdict(cfg),
            "best_review": best_review,
            "best_score": best_score.to_dict(),
            "history": history,
        }
