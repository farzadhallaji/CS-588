"""
Generate multiple candidate reviews per instance using existing editors.
Outputs JSONL records that keep the same instance schema used by evaluate.py.
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

from core.data import build_instances, load_raw_data, split_by_language  # type: ignore
from core.editors import HFLocalEditor, OllamaEditor, EchoEditor  # type: ignore
from core.evidence import EvidenceRetriever  # type: ignore
from core.loop import LoopConfig  # type: ignore

from .prompts import build_prompt, PROMPT_VARIANTS


def choose_editor(args: argparse.Namespace):
    if args.model_type == "ollama":
        editor = OllamaEditor(model=args.model_name, temperature=args.temperature)
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
    if args.model_type == "echo":
        return EchoEditor()
    raise ValueError(f"Unknown model_type {args.model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate best-of-N candidates for review_reward_rerank.")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=2, help="Number of generations per prompt variant.")
    parser.add_argument("--prompt-variants", type=str, default="default,evidence_grounded,test_heavy,concise")
    parser.add_argument("--model-type", choices=["ollama", "hf-local", "echo"], default="ollama")
    parser.add_argument("--model-name", type=str, default="llama3:8b-instruct-q4_0")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="Stored for downstream scoring consistency.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "candidates.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_variants = [p.strip() for p in args.prompt_variants.split(",") if p.strip()]
    for pv in prompt_variants:
        if pv not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant '{pv}'. Valid: {PROMPT_VARIANTS}")

    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    if args.limit:
        selected = selected[: args.limit]

    editor = choose_editor(args)
    retriever = EvidenceRetriever()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as f:
        for inst in selected:
            evidence_map = retriever.retrieve(inst.pseudo_refs, inst.patch, inst.old_file)
            evidence_lines = [ln for lines in evidence_map.values() for ln in lines]
            record = {
                "instance": {"idx": inst.idx, "lang": inst.lang, "meta": inst.meta},
                "seed_review": inst.review,
                "claims": inst.pseudo_refs,
                "patch": inst.patch,
                "old_file": inst.old_file,
                "evidence": evidence_map,
                "candidates": [],
                "tau": args.tau,
            }
            for pv in prompt_variants:
                prompt_text = build_prompt(
                    variant=pv,
                    seed_review=inst.review,
                    claims=inst.pseudo_refs,
                    diff=inst.patch,
                    old_code=inst.old_file,
                    uncovered_claims=[],
                    offending_sentences=[],
                    evidence_snippets=evidence_lines,
                )
                samples = editor.propose(
                    current_review=inst.review,
                    uncovered=[],
                    offending=[],
                    evidence=evidence_map,
                    prompt=prompt_text,
                    num_samples=args.num_samples,
                )
                for text in samples:
                    record["candidates"].append(
                        {
                            "text": text.strip(),
                            "prompt_variant": pv,
                            "temperature": args.temperature,
                            "prompt": prompt_text,
                            "model_type": args.model_type,
                            "model_name": args.model_name,
                        }
                    )
            f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
