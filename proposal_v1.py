"""
Few-shot baseline mirroring the initial proposal: degrade good reviews -> improve with CRScore cues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import build_instances, load_raw_data, split_by_language
from core.scoring import CRScorer
from core.editors import OllamaEditor
from core.loop import BASE_SYSTEM_PROMPT


def build_prompt(pairs, score_dict, diff: str, review: str) -> str:
    shots = []
    for ex in pairs:
        shots.append("Example (low quality):\n" + ex["bad"])
        shots.append("Example (improved):\n" + ex["good"])
    shots_block = "\n\n".join(shots)

    return f"""{shots_block}

A quality assessment has been performed (0..1):
Comprehensiveness, Relevance, Conciseness.
Use lower dimensions to guide improvement.

Scores: {score_dict}

Code change (diff):
{diff}

Junior review:
{review}

Use the examples to transform the junior review into a higher-quality review.
Use the score dimensions to guide edits:
- If Comp is low: add missing key points from the change.
- If Con is low: delete vague/unrelated sentences.
- If Rel is low: align with the actual change and implications.
Rules:
- Do not invent details not supported by the diff or evidence implied by the examples.
- Prefer 1-4 high-signal sentences.

Improved review:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proposal v1 few-shot baseline.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json",
        help="Path to raw_data.json.",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--tau", type=float, default=0.6, help="CRScore similarity threshold.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="SentenceTransformer model path for scoring.",
    )
    parser.add_argument(
        "--fewshot",
        type=Path,
        default=ROOT / "results" / "fewshot_pairs.json",
        help="Few-shot pairs JSON produced by scripts/build_fewshot_pairs.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "proposal_v1.jsonl",
        help="Where to write baseline outputs.",
    )
    parser.add_argument("--ollama-model", type=str, default="llama3:8b-instruct-q4_0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fewshot.exists():
        raise FileNotFoundError(f"Missing few-shot pairs: {args.fewshot}. Run scripts/build_fewshot_pairs.py first.")
    pairs = json.loads(args.fewshot.read_text())

    scorer = CRScorer(model_path=args.model_path, tau=args.tau)
    editor = OllamaEditor(model=args.ollama_model, temperature=0.2)
    editor.system = BASE_SYSTEM_PROMPT

    instances = build_instances(load_raw_data(args.raw_data))
    splits = split_by_language(instances)
    selected = splits[args.split]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for inst in selected:
            seed = inst.review.strip()
            seed_score = scorer.score(inst.pseudo_refs, seed).to_dict()
            prompt = build_prompt(pairs, seed_score, inst.patch, seed)
            improved = editor.propose(seed, [], [], {}, prompt, num_samples=1)[0].strip() or seed
            final_score = scorer.score(inst.pseudo_refs, improved).to_dict()
            f.write(
                json.dumps(
                    {
                        "instance": {"idx": inst.idx, "lang": inst.lang, "meta": inst.meta},
                        "seed_review": seed,
                        "seed_score": seed_score,
                        "best_review": improved,
                        "best_score": final_score,
                        "method": "proposal_v1_fewshot_singlepass",
                        "tau": args.tau,
                    }
                )
                + "\n"
            )
            f.flush()
            print(
                f"[{inst.lang} idx={inst.idx}] seed Rel={seed_score['Rel']:.3f} -> "
                f"{final_score['Rel']:.3f}"
            )


if __name__ == "__main__":
    main()
