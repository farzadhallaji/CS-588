from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ReviewInstance:
    idx: int
    lang: str
    review: str
    patch: str
    old_file: str
    pseudo_refs: List[str]
    meta: Dict[str, str] = field(default_factory=dict)


def load_raw_data(path: Path) -> List[dict]:
    raw = path.read_text().strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def extract_pseudo_refs(rec: dict) -> List[str]:
    pseudo_refs: List[str] = []
    claims = rec.get("claims") or []
    for item in claims:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pseudo_refs.append(str(item[1]).strip())
        elif isinstance(item, str):
            pseudo_refs.append(item.strip())
    return [p for p in pseudo_refs if p]


def build_instances(records: List[dict]) -> List[ReviewInstance]:
    instances: List[ReviewInstance] = []
    for rec in records:
        idx_val = rec.get("index", rec.get("idx"))
        try:
            inst_idx = int(idx_val) if idx_val is not None else len(instances)
        except (TypeError, ValueError):
            inst_idx = len(instances)
        refs = extract_pseudo_refs(rec)
        instances.append(
            ReviewInstance(
                idx=inst_idx,
                lang=str(rec.get("lang", "unk")),
                review=str(rec.get("msg", "")).strip(),
                patch=str(rec.get("patch", "")),
                old_file=str(rec.get("oldf", "")),
                pseudo_refs=refs,
                meta={
                    "id": str(rec.get("id", "")),
                    "proj": str(rec.get("proj", "")),
                    "index": str(rec.get("index", "")),
                },
            )
        )
    return instances


def split_by_language(
    instances: List[ReviewInstance], dev_per_lang: int = 60, test_per_lang: int = 40, seed: int = 13
) -> Dict[str, List[ReviewInstance]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[ReviewInstance]] = {}
    for inst in instances:
        grouped.setdefault(inst.lang, []).append(inst)
    dev, test = [], []
    for lang in sorted(grouped):
        rows = grouped[lang][:]
        rng.shuffle(rows)
        dev.extend(rows[:dev_per_lang])
        test.extend(rows[dev_per_lang : dev_per_lang + test_per_lang])
    return {"dev": dev, "test": test}
