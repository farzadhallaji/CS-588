"""
Quick smoke test for the ProposedApproach loop.

Runs one instance from the dev split with the offline template editor to verify
data loading, scoring, evidence retrieval, and the iterative loop wire-up.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.scoring import CRScorer
from core.loop import IterativeRefiner, LoopConfig
from core.editors import TemplateEditor
from core.data import build_instances, load_raw_data, split_by_language


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "CRScore" / "human_study" / "phase1" / "raw_data.json"
    if not raw_path.exists():
        print(f"Missing dataset at {raw_path}; please ensure CRScore data is present.")
        sys.exit(1)

    records = load_raw_data(raw_path)
    instances = build_instances(records)
    splits = split_by_language(instances)
    sample = splits["dev"][0]

    try:
        scorer = CRScorer(tau=0.6)
    except Exception as exc:  # pragma: no cover - sanity print for missing deps
        print(f"Dependency error: {exc}")
        print("Install deps via: pip install sentence-transformers torch")
        sys.exit(1)

    editor = TemplateEditor()
    cfg = LoopConfig(max_iter=2, num_samples=1, tau=0.6, tau_evidence=0.35, max_sentence_change=0.7)
    runner = IterativeRefiner(scorer, editor, cfg)
    result = runner.run(sample)
    print("Claims/pseudo-refs:", sample.pseudo_refs[:3])
    print("Initial review:", sample.review)
    print("Best score:", result["best_score"])
    print("Best review:", result["best_review"])


if __name__ == "__main__":
    main()
