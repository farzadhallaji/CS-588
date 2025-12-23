"""
End-to-end runner:
1) generate candidates (optionally with a LoRA adapter), or load pre-generated candidates,
2) select the best review using the reward function,
3) write evaluation-compatible JSONL (instance + best_review).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import numpy as np

from . import PROPOSED_ROOT

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

from core.data import build_instances, load_raw_data, split_by_language  # type: ignore
from core.editors import HFLocalEditor, OllamaEditor, EchoEditor, BaseEditor  # type: ignore
from core.evidence import EvidenceRetriever  # type: ignore
from core.loop import LoopConfig  # type: ignore
from core.scoring import CRScorer  # type: ignore
from core.utils import sentence_split  # type: ignore

from .prompts import build_prompt, PROMPT_VARIANTS
from .soft_crscore import SoftCRScoreResult, embed_texts, soft_crscore
from .evidence_penalty import collect_evidence, flatten_evidence_map, evidence_penalty
from .reward import RewardWeights, compute_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="review_reward_rerank inference entry point.")
    parser.add_argument("--mode", choices=["select", "generate"], default="select", help="select: use provided candidates; generate: create then select.")
    parser.add_argument("--candidates", type=Path, help="Existing candidates JSONL (required for mode=select).")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=2, help="Generations per prompt variant when mode=generate.")
    parser.add_argument("--prompt-variants", type=str, default="default,evidence_grounded,test_heavy,concise")
    parser.add_argument("--model-type", choices=["ollama", "hf-local", "echo", "lora"], default="ollama")
    parser.add_argument("--model-name", type=str, default="llama3:8b-instruct-q4_0", help="Ollama model name or HF path.")
    parser.add_argument("--lora-path", type=Path, default=None, help="Path to LoRA adapter (used when --model-type lora).")
    parser.add_argument("--lora-base", type=str, default=None, help="Base model name for LoRA (defaults to --model-name).")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load HF base in 4-bit (LoRA).")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="SentenceTransformer model for scoring.")
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="CRScore threshold; keep consistent with evaluate.py.")
    parser.add_argument("--temp", type=float, default=0.05, help="SoftCRScore temperature.")
    parser.add_argument("--evidence-margin", type=float, default=0.35)
    parser.add_argument("--top-k-align", type=int, default=2)
    parser.add_argument("--score-mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--w-rel", type=float, default=1.0)
    parser.add_argument("--w-unsupported", type=float, default=0.6)
    parser.add_argument("--w-len", type=float, default=0.02)
    parser.add_argument("--w-copy", type=float, default=0.15)
    parser.add_argument("--len-norm", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "final_outputs.jsonl")
    parser.add_argument("--save-candidates", type=Path, default=None, help="Optional path to dump generated candidates for reuse.")
    return parser.parse_args()


def choose_editor(args: argparse.Namespace) -> BaseEditor:
    if args.model_type == "ollama":
        return OllamaEditor(model=args.model_name, temperature=args.temperature)
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
    if args.model_type == "lora":
        if not args.lora_path:
            raise ValueError("Provide --lora-path when using --model-type lora.")
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install transformers + peft to use --model-type lora.") from exc
        base_model = args.lora_base or args.model_name
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=args.load_in_4bit,
            device_map="auto" if args.device != "cpu" else None,
        )
        model = PeftModel.from_pretrained(model, args.lora_path or base_model)
        tok = AutoTokenizer.from_pretrained(base_model)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device_map="auto" if args.device != "cpu" else None,
            return_full_text=False,
        )

        class LoraEditor(BaseEditor):
            def propose(self, current_review, uncovered, offending, evidence, prompt, num_samples):
                outputs = gen(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=num_samples,
                    pad_token_id=tok.eos_token_id,
                )
                texts = []
                for out in outputs:
                    text = out.get("generated_text", "").strip()
                    if "\n\n" in text:
                        text = text.split("\n\n")[-1].strip()
                    texts.append(text)
                return texts

        return LoraEditor()
    raise ValueError(f"Unknown model_type {args.model_type}")


def generate_candidates(instances, args: argparse.Namespace, prompt_variants: List[str]) -> List[dict]:
    editor = choose_editor(args)
    retriever = EvidenceRetriever()
    records = []
    for inst in instances:
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
        records.append(record)
    return records


def hard_to_soft(score, top_k: int) -> SoftCRScoreResult:
    sim = score.sim_matrix.T  # sentences x claims
    alignments: List[List[dict]] = []
    for i in range(sim.shape[0]):
        row = sim[i]
        if row.size == 0:
            alignments.append([])
            continue
        top_idx = np.argsort(-row)[: max(top_k, 1)]
        alignments.append([{"claim_idx": int(j), "score": float(row[j])} for j in top_idx])
    return SoftCRScoreResult(
        soft_precision=score.conciseness,
        soft_recall=score.comprehensiveness,
        soft_f1=score.relevance,
        alignments=alignments,
        sim_matrix=sim,
    )


def score_and_select(rec: dict, scorer: CRScorer, args: argparse.Namespace, weights: RewardWeights) -> dict:
    claims = rec.get("claims", []) or []
    claim_embs = embed_texts(scorer.sbert, claims)
    patch = rec.get("patch", "")
    old_file = rec.get("old_file", "")
    evidence_map = rec.get("evidence") or collect_evidence(claims, patch, old_file)
    evidence_lines = flatten_evidence_map(evidence_map)
    candidates = rec.get("candidates", []) or []
    seed = rec.get("seed_review", "")

    if not candidates:
        return {
            "instance": rec.get("instance", {}),
            "seed_review": seed,
            "best_review": seed,
            "selection": {
                "reward": None,
                "soft_scores": None,
                "unsupported_rate": None,
                "chosen_from": 0,
            },
            "alignments": [],
            "evidence_map": evidence_map,
        }

    scored = []
    for cand in candidates:
        text = cand.get("text", "").strip() or seed
        sentences = sentence_split(text)
        if args.score_mode == "hard":
            hard = scorer.score(claims, text)
            soft_res = hard_to_soft(hard, args.top_k_align)
        else:
            sent_embs = embed_texts(scorer.sbert, sentences)
            soft_res = soft_crscore(sent_embs, claim_embs, tau=args.tau, temp=args.temp, top_k=args.top_k_align)
        evidence_res = evidence_penalty(scorer, sentences, evidence_lines, margin=args.evidence_margin)
        reward_bd = compute_reward(soft_res, evidence_res, text, claims, weights)
        alignments = []
        for i, sent in enumerate(sentences):
            aligns = soft_res.alignments[i] if i < len(soft_res.alignments) else []
            alignments.append({"sentence": sent, "claims": aligns})
        scored.append(
            {
                "text": text,
                "reward": reward_bd.to_dict(),
                "soft_scores": soft_res.to_dict(),
                "evidence": evidence_res.to_dict(),
                "alignments": alignments,
            }
        )

    best = max(scored, key=lambda x: x["reward"]["reward"])
    return {
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

    prompt_variants = [p.strip() for p in args.prompt_variants.split(",") if p.strip()]
    for pv in prompt_variants:
        if pv not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown prompt variant '{pv}'. Valid: {PROMPT_VARIANTS}")

    candidate_records: List[dict] = []
    if args.mode == "select":
        if not args.candidates:
            raise ValueError("Provide --candidates when mode=select.")
        with args.candidates.open() as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    break
                if not line.strip():
                    continue
                candidate_records.append(json.loads(line))
    else:
        records = load_raw_data(args.raw_data)
        instances = build_instances(records)
        splits = split_by_language(instances)
        selected = instances if args.split == "all" else splits[args.split]
        if args.limit:
            selected = selected[: args.limit]
        candidate_records = generate_candidates(selected, args, prompt_variants)
        if args.save_candidates:
            args.save_candidates.parent.mkdir(parents=True, exist_ok=True)
            with args.save_candidates.open("w") as f:
                for rec in candidate_records:
                    f.write(json.dumps(rec) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fout:
        for rec in candidate_records:
            selected = score_and_select(rec, scorer, args, weights)
            fout.write(json.dumps(selected) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
