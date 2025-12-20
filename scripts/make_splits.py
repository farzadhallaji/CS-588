"""
Export deterministic dev/test splits for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.data import build_instances, load_raw_data, split_by_language


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist dev/test split indices.")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=ROOT.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json",
        help="Path to raw_data.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "splits.json",
        help="Where to write the split indices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_raw_data(args.raw_data)
    instances = build_instances(records)
    splits = split_by_language(instances)

    payload = {
        "dev": [{"idx": inst.idx, "lang": inst.lang, "id": inst.meta.get("id", "")} for inst in splits["dev"]],
        "test": [{"idx": inst.idx, "lang": inst.lang, "id": inst.meta.get("id", "")} for inst in splits["test"]],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
