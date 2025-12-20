"""
Main entry: CRScore-guided iterative refinement loop with ablations.
Self-contained implementation (no CRScore package imports) for publication.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import build_instances, load_raw_data, split_by_language
from core.editors import EchoEditor, HFLocalEditor, OllamaEditor, TemplateEditor
from core.loop import IterativeRefiner, LoopConfig
from core.scoring import CRScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the iterative refinement loop.")
    parser.add_argument("--raw-data", type=Path, default=Path(__file__).resolve().parent.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="SentenceTransformer model path or HF id.")
    parser.add_argument("--tau", type=float, default=LoopConfig.tau, help="Similarity threshold for CRScore scoring.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--lang", choices=["py", "java", "js", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "outputs.jsonl")
    parser.add_argument("--mode", choices=["loop", "k1", "no-selection", "no-evidence", "rewrite"], default="loop")
    parser.add_argument("--editor-type", choices=["template", "ollama", "echo", "hf-local"], default="template")
    parser.add_argument("--ollama-model", type=str, default="llama3:8b-instruct-q4_0")
    parser.add_argument("--hf-model-path", type=str, default=None, help="Local HF model path/name for hf-local editor.")
    parser.add_argument("--hf-max-new-tokens", type=int, default=128)
    parser.add_argument("--hf-temperature", type=float, default=0.6)
    parser.add_argument("--hf-top-p", type=float, default=0.9)
    parser.add_argument("--hf-device", type=str, default="cpu", help="Device for HF local model (cpu, cuda, etc.).")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--tau-evidence", type=float, default=LoopConfig.tau_evidence, help="Similarity threshold for evidence support.")
    parser.add_argument("--max-change", type=float, default=LoopConfig.max_sentence_change, help="Maximum allowed sentence change ratio.")
    return parser.parse_args()


def build_editor(args: argparse.Namespace):
    if args.editor_type == "ollama":
        return OllamaEditor(model=args.ollama_model)
    if args.editor_type == "echo":
        return EchoEditor()
    if args.editor_type == "hf-local":
        if not args.hf_model_path:
            raise ValueError("--hf-model-path is required for hf-local (LLM). --model-path is for embeddings.")
        model_path = args.hf_model_path
        return HFLocalEditor(
            model_path=model_path,
            max_new_tokens=args.hf_max_new_tokens,
            temperature=args.hf_temperature,
            top_p=args.hf_top_p,
            device=args.hf_device,
        )
    return TemplateEditor()


def config_from_args(args: argparse.Namespace) -> LoopConfig:
    cfg = LoopConfig(max_iter=args.max_iter, num_samples=args.num_samples, tau=args.tau)
    cfg.tau_evidence = args.tau_evidence
    cfg.max_sentence_change = args.max_change
    if args.mode == "k1":
        cfg.max_iter = 1
    if args.mode == "no-selection":
        cfg.selection = "random"
    if args.mode == "no-evidence":
        cfg.disable_evidence = True
    if args.mode == "rewrite":
        cfg.rewrite = True
        cfg.max_sentence_change = 1.0
    return cfg


def main() -> None:
    random.seed(0)
    args = parse_args()
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)
    selected = instances if args.split == "all" else splits[args.split]
    if args.lang != "all":
        selected = [inst for inst in selected if inst.lang == args.lang]
    if args.limit:
        selected = selected[: args.limit]

    scorer = CRScorer(model_path=args.model_path, tau=args.tau)
    editor = build_editor(args)
    cfg = config_from_args(args)
    runner = IterativeRefiner(scorer, editor, cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for inst in selected:
            result = runner.run(inst)
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(
                f"[{inst.lang} idx={inst.idx}] Rel={result['best_score']['Rel']:.3f} "
                f"Con={result['best_score']['Con']:.3f} Comp={result['best_score']['Comp']:.3f}"
            )


if __name__ == "__main__":
    main()
