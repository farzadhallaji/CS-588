from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import List, Sequence


def sentence_split(text: str) -> List[str]:
    parts: List[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(chunk if chunk.endswith((".", "!", "?")) else chunk + ".")
    return parts or [text.strip()]


def lexical_overlap(a: str, b: str) -> int:
    atok = set(a.lower().split())
    btok = set(b.lower().split())
    return len(atok & btok)


def sentence_change_ratio(old: Sequence[str], new: Sequence[str]) -> float:
    if not old and not new:
        return 0.0
    sm = SequenceMatcher(None, list(old), list(new))
    return 1.0 - sm.ratio()
