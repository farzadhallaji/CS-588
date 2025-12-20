from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


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
    seen_idxs: Set[int] = set()

    def _coerce_int(val, default: int | None) -> int | None:
        try:
            return int(val)
        except (TypeError, ValueError):
            try:
                return int(str(val).strip())
            except Exception:
                return default

    for i, rec in enumerate(records):
        # Prefer provided idx/index/id; fall back to positional index.
        inst_idx = None
        for candidate in (rec.get("idx"), rec.get("index"), rec.get("id")):
            if candidate is None or candidate == "":
                continue
            coerced = _coerce_int(candidate, default=None)
            if coerced is not None:
                inst_idx = coerced
                break
        if inst_idx is None:
            inst_idx = i

        # Deduplicate to avoid collisions that would break evaluation mapping.
        if inst_idx in seen_idxs:
            inst_idx = max(seen_idxs) + 1
            while inst_idx in seen_idxs:
                inst_idx += 1
        seen_idxs.add(inst_idx)

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
