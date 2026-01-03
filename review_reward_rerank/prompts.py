"""
Prompt bank for review_reward_rerank experiments.

Templates are plain Python strings with simple `.format()` placeholders.
Inputs:
  seed_review, claims, diff, old_code, uncovered_claims, offending_sentences, evidence_snippets
Output:
  A single review text (no extra formatting).
"""

from __future__ import annotations

from textwrap import dedent
from typing import Dict, Iterable, List

PROMPT_TEMPLATES: Dict[str, str] = {
    "default": dedent(
        """
        You are improving a code review comment.

        Seed review:
        {seed_review}

        Pseudo-references (claims about what should be mentioned):
        {claims}

        Task:
        Rewrite the review so it covers the important claims, stays concise, and does not add new unverifiable claims.
        Return ONLY the rewritten review text.
        """
    ).strip(),
    "concise": dedent(
        """
        Rewrite the seed review into at most 3 bullet points.
        Cover as many claims as possible without adding new information.
        No greetings. No filler. Return ONLY the bullets.

        Seed review:
        {seed_review}

        Claims:
        {claims}
        """
    ).strip(),
    "evidence_grounded": dedent(
        """
        Rewrite the review, but every statement must be grounded in the provided diff evidence.
        If something is uncertain, phrase it as a question or suggest a verification/test instead of asserting it.

        Seed review:
        {seed_review}

        Claims:
        {claims}

        Diff evidence:
        {diff}

        (Optionally old code context)
        {old_code}

        Uncovered claims to prioritize:
        {uncovered_claims}

        Offending sentences to trim or rewrite:
        {offending_sentences}

        Return ONLY the rewritten review.
        """
    ).strip(),
    "test_heavy": dedent(
        """
        Rewrite the review with a verification mindset:
        - If a claim cannot be directly verified from the diff, convert it into a test suggestion or a question.
        - Prefer actionable checks (unit tests, edge cases, logging, invariants).

        Seed review:
        {seed_review}

        Claims:
        {claims}

        Diff:
        {diff}

        Evidence snippets (optional):
        {evidence_snippets}

        Return ONLY the rewritten review.
        """
    ).strip(),
    "review_enhancer_strict": dedent(
        r"""
        You enhance (NOT rewrite from scratch) a human code review comment.

        HARD RULES (do not break these):
        - Preserve intent and stance from the seed review.
          * If the seed is unsure/hedged, keep it unsure/hedged.
          * If the seed asks a question, KEEP IT A QUESTION (must contain '?' and remain a question).
          * Do not turn questions/suggestions into factual claims.
        - Keep it short and high-signal.
          * Target 2–4 bullets OR 3–6 short sentences (choose the shorter that still covers the important points).
          * Max ~90 words total. No greetings. No filler.
        - No new unverifiable claims.
          * Only use what is supported by Claims and (if provided) Diff/Evidence.
          * If something is not verifiable, phrase it as a question or a test/verification suggestion.
        - Cover important points.
          * Prioritize Uncovered claims first, then remaining claims.
          * Remove or rewrite Offending sentences, but keep any important intent they contain.

        INPUTS
        [SEED REVIEW]
        {seed_review}

        [CLAIMS TO COVER]
        {claims}

        [UNCOVERED CLAIMS (PRIORITY)]
        {uncovered_claims}

        [OFFENDING SENTENCES (TRIM/REWRITE)]
        {offending_sentences}

        [DIFF (EVIDENCE, OPTIONAL)]
        {diff}

        [EVIDENCE SNIPPETS (OPTIONAL)]
        {evidence_snippets}

        [OLD CODE CONTEXT (OPTIONAL)]
        {old_code}

        OUTPUT REQUIREMENTS
        - Return ONLY the improved review text.
        - Keep questions as questions.
        - Be concise and actionable.
        - Do NOT mention these rules.
        """
    ).strip(),
    "review_enhancer_strict_fewshot": dedent(
        r"""
        Task: Enhance a code review comment.
        IMPORTANT: Preserve the seed's intent. Keep questions as questions. Do not invent claims.
        Keep it short (max ~90 words). No greetings. No filler.

        Allowed edits: rephrase, reorder, merge, delete fluff, convert unverifiable assertions into questions/tests.
        Forbidden: new facts not supported by Claims/Diff/Evidence; changing a question into a statement.

        Example:
        Seed: "Not sure this handles null. Can you add a check? Also tests?"
        Claims: - null handling  - add tests
        Output:
        - Does this handle null inputs? Consider an explicit guard/check.
        - Add a unit test covering null/empty edge cases.

        Now do the same for the following.

        [SEED REVIEW]
        {seed_review}

        [CLAIMS TO COVER]
        {claims}

        [UNCOVERED CLAIMS (PRIORITY)]
        {uncovered_claims}

        [OFFENDING SENTENCES (TRIM/REWRITE)]
        {offending_sentences}

        [DIFF (EVIDENCE, OPTIONAL)]
        {diff}

        [EVIDENCE SNIPPETS (OPTIONAL)]
        {evidence_snippets}

        [OLD CODE CONTEXT (OPTIONAL)]
        {old_code}

        Output ONLY the improved review text.
        """
    ).strip(),
    "claim_extraction": dedent(
        """
        Given this code diff, list the key review-worthy claims (max 6) as short, atomic items.
        Do NOT invent changes not present in the diff.
        Format: one claim per line.

        Diff:
        {diff}
        """
    ).strip(),
}

PROMPT_VARIANTS = [
    "default",
    "concise",
    "evidence_grounded",
    "test_heavy",
    "review_enhancer_strict",
    "review_enhancer_strict_fewshot",
]


def _fmt_block(title: str, items: Iterable[str]) -> str:
    values = [str(x).strip() for x in items if str(x).strip()]
    if not values:
        return f"{title}: (none provided)"
    joined = "\n".join(f"- {v}" for v in values)
    return f"{title}:\n{joined}"


def build_prompt(
    variant: str,
    seed_review: str,
    claims: List[str],
    diff: str = "",
    old_code: str = "",
    uncovered_claims: List[str] | None = None,
    offending_sentences: List[str] | None = None,
    evidence_snippets: List[str] | None = None,
) -> str:
    """Return a formatted prompt for the requested variant."""
    if variant not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt variant '{variant}'. Options: {list(PROMPT_TEMPLATES)}")

    claims_text = "\n".join(f"- {c}" for c in claims) if claims else "- None provided."
    prompt = PROMPT_TEMPLATES[variant].format(
        seed_review=seed_review or "(empty)",
        claims=claims_text,
        diff=diff or "(diff unavailable)",
        old_code=old_code or "(not provided)",
        uncovered_claims="\n".join(uncovered_claims or []) or "(not specified)",
        offending_sentences="\n".join(offending_sentences or []) or "(none)",
        evidence_snippets="\n".join(evidence_snippets or []) or "(not provided)",
    )
    return prompt.strip()
