"""
Helpers to decide whether to skip/rerun experiment outputs based on completeness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.data import build_instances, load_raw_data, split_by_language


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for _ in fh)


def expected_instances(raw_path: Path, split: str, lang: str | None = None, limit: int | None = None) -> int:
    records = load_raw_data(raw_path)
    instances = build_instances(records)
    if split != "all":
        instances = split_by_language(instances)[split]
    if lang and lang != "all":
        instances = [inst for inst in instances if inst.lang == lang]
    if limit:
        instances = instances[:limit]
    return len(instances)


def should_skip_output(
    output_path: Path, expected: Optional[int], force: bool = False, label: str = "output"
) -> bool:
    """
    Return True if the output can be skipped (already complete).
    - If force is True, never skip.
    - If expected is provided, skip when existing lines >= expected.
    - If expected is None, skip when file exists.
    """
    if force:
        return False
    if output_path.exists():
        lines = count_lines(output_path)
        if expected is None:
            print(f"Skipping {label}; found existing file {output_path}")
            return True
        if lines >= expected:
            print(f"Skipping {label}; found {lines}/{expected} rows in {output_path}")
            return True
        print(f"Rerunning {label}; found {lines}/{expected} rows (incomplete)")
    return False
