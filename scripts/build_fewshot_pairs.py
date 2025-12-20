"""
Construct few-shot bad->good pairs for proposal v1 baseline using dev split only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import build_instances, load_raw_data, split_by_language
from core.scoring import CRScorer
from core.utils import sentence_split


def degrade(review: str) -> str:
    sents = sentence_split(review)
    if len(sents) <= 1:
        return review.strip()
    k = max(1, int(len(sents) * random.uniform(0.3, 0.6)))
    kept = random.sample(sents, k=k)
    if random.random() < 0.5:
        kept.append("Looks good overall.")
    return " ".join(kept).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build degraded few-shot pairs from the dev split.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json",
        help="Path to raw_data.json.",
    )
    parser.add_argument("--tau", type=float, default=0.7314, help="CRScore similarity threshold.")
    parser.add_argument("--model-path", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="SentenceTransformer model.")
    parser.add_argument("--max-pairs", type=int, default=12, help="How many few-shot pairs to keep.")
    parser.add_argument("--candidate-pool", type=int, default=60, help="Top-N high-quality dev reviews to consider.")
    parser.add_argument("--max-refs", type=int, default=8, help="Max pseudo-refs to keep per example.")
    parser.add_argument("--bad-rel-thresh", type=float, default=0.35, help="Upper bound on relevance for degraded review.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "fewshot_pairs.json",
        help="Where to write the constructed pairs.",
    )
    return parser.parse_args()


def main() -> None:
    random.seed(0)
    args = parse_args()
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    dev_split = split_by_language(instances)["dev"]

    scorer = CRScorer(model_path=args.model_path, tau=args.tau)
    scored = []
    for inst in dev_split:
        rel = scorer.score(inst.pseudo_refs, inst.review).relevance
        scored.append((rel, inst))
    scored.sort(key=lambda x: x[0], reverse=True)

    pairs = []
    for _, inst in scored[: args.candidate_pool]:
        good = inst.review.strip()
        bad = degrade(good)
        for _ in range(5):
            if scorer.score(inst.pseudo_refs, bad).relevance < args.bad_rel_thresh:
                break
            bad = degrade(good)
        pairs.append(
            {
                "lang": inst.lang,
                "good": good,
                "bad": bad,
                "pseudo_refs": inst.pseudo_refs[: args.max_refs],
            }
        )
        if len(pairs) >= args.max_pairs:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pairs, indent=2))
    print(f"Wrote {args.out} with {len(pairs)} pairs")


if __name__ == "__main__":
    main()
