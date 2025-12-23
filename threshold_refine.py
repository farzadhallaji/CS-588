"""
Threshold-gated local LLM refinement (first proposal variant).

Algorithm:
- Compute CRScore(Rel) of the seed review.
- If Rel < threshold -> call a local LLM once with a prompt variant to refine.
- Otherwise keep the original review.

Supports multiple prompt variants and local backends (Ollama or HF local).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import ReviewInstance, build_instances, load_raw_data, split_by_language
from core.scoring import CRScorer
from core.editors import HFLocalEditor, OllamaEditor
from core.loop import LoopConfig, BASE_SYSTEM_PROMPT
from core.checks import expected_instances, should_skip_output


PROMPTS: Dict[str, str] = {
    "default": (
        "Improve the review to better match the claims.\n"
        "Include missing important points, remove irrelevant content.\n"
        "If a claim is uncertain, phrase it as a verification/test request.\n"
        "Output only the revised review."
    ),
    "concise": (
        "Rewrite the review into 1-3 sentences.\n"
        "Must mention the main change and one concrete check/test.\n"
        "Remove everything else.\n"
        "Output only the revised review."
    ),
    "evidence": (
        "Add ONLY points that are directly supported by the claims text.\n"
        "If support is missing, do not add the point (suggest a test instead).\n"
        "Keep it short and specific.\n"
        "Output only the revised review."
    ),
    "test-heavy": (
        "Focus on test coverage and failure modes implied by the claims.\n"
        "Add 1-2 concrete tests (happy path + edge case).\n"
        "Avoid restating obvious change details unless needed.\n"
        "Output only the revised review."
    ),
}


def build_prompt(review: str, claims: List[str], prompt_variant: str) -> str:
    base = PROMPTS.get(prompt_variant, PROMPTS["default"])
    claims_block = "\n".join(f"- {c}" for c in claims)
    return f"{base}\n\nCurrent review:\n{review}\n\nClaims:\n{claims_block}\n\nRevised review:"


def choose_editor(args: argparse.Namespace):
    if args.model_type == "ollama":
        editor = OllamaEditor(model=args.model_name, temperature=args.temperature)
        editor.system = BASE_SYSTEM_PROMPT
        return editor
    if args.model_type == "hf-local":
        if not args.model_name:
            raise ValueError("Provide --model-name for hf-local (path or HF id).")
        return HFLocalEditor(
            model_path=args.model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
    raise ValueError(f"Unknown model_type {args.model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold-gated local LLM refinement with CRScore.")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="CRScore threshold for scoring.")
    parser.add_argument("--threshold", type=float, default=0.6, help="If Rel < threshold, trigger refinement.")
    parser.add_argument("--model-type", choices=["ollama", "hf-local"], default="ollama")
    parser.add_argument("--model-name", type=str, default="llama3:8b-instruct-q4_0", help="Ollama model name or HF path.")
    parser.add_argument("--prompt-variant", choices=list(PROMPTS.keys()), default="default")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "threshold_refine.jsonl")
    parser.add_argument("--expected-count", type=int, default=None, help="Override expected number of rows; defaults to dataset split size (after limit).")
    parser.add_argument("--force", action="store_true", help="Do not skip even if output already complete.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = args.expected_count or expected_instances(args.raw_data, args.split, limit=args.limit)
    if should_skip_output(args.output, expected, args.force, label="threshold_refine output"):
        return
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    if args.limit:
        selected = selected[: args.limit]

    scorer = CRScorer(tau=args.tau)
    editor = choose_editor(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for inst in selected:
            # score seed
            seed_score = scorer.score(inst.pseudo_refs, inst.review)
            improved = False
            best_review = inst.review
            final_score = seed_score

            if seed_score.relevance < args.threshold:
                prompt = build_prompt(inst.review, inst.pseudo_refs, args.prompt_variant)
                cand = editor.propose(
                    current_review=inst.review,
                    uncovered=[],
                    offending=[],
                    evidence={},
                    prompt=prompt,
                    num_samples=1,
                )[0]
                best_review = cand.strip() or inst.review
                final_score = scorer.score(inst.pseudo_refs, best_review)
                improved = final_score.relevance > seed_score.relevance

            f.write(
                json.dumps(
                    {
                        "instance": {"idx": inst.idx, "lang": inst.lang, "meta": inst.meta},
                        "seed_score": seed_score.to_dict(),
                        "final_score": final_score.to_dict(),
                        "seed_review": inst.review,
                        "best_review": best_review,
                        "improved": improved,
                        "prompt_variant": args.prompt_variant,
                        "threshold": args.threshold,
                        "model_type": args.model_type,
                        "model_name": args.model_name,
                    }
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
