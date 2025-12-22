"""
Prompt bank for FinalProposal experiments.

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

PROMPT_VARIANTS = ["default", "concise", "evidence_grounded", "test_heavy"]


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
