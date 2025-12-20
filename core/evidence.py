from __future__ import annotations

from typing import Dict, List, Tuple

from .utils import lexical_overlap


# Allow optional ,len in hunks
HUNK_RE = r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"


def generate_newf(oldf: str, diff: str) -> Tuple[str, Tuple[int, int]]:
    """
    Rebuild new file content from a unified diff (single hunk supported).
    Returns (new_file, (start_line, end_line)) for the changed span.
    """
    import re

    oldflines = oldf.split("\n")
    difflines = diff.split("\n")
    if not difflines:
        return "", (-1, -1)

    hunk_i = next((i for i, ln in enumerate(difflines) if ln.startswith("@@ ")), None)
    if hunk_i is None:
        return "", (-1, -1)
    first_line = difflines[hunk_i]
    body = [ln for ln in difflines[hunk_i + 1 :] if ln != r"\ No newline at end of file"]

    matchres = re.match(HUNK_RE, first_line)
    if not matchres:
        return "", (-1, -1)
    old_start, old_len, _, new_len = matchres.groups()
    old_start = int(old_start) - 1
    old_len = int(old_len) if old_len is not None else 1
    new_len = int(new_len) if new_len is not None else 1

    prevlines = oldflines[:old_start]
    afterlines = oldflines[old_start + old_len :]
    lines: List[str] = []

    for line in body:
        if line.startswith("@@ "):
            break  # stop at next hunk (single-hunk support)
        if line.startswith("-"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])

    merged_lines = prevlines + lines + afterlines
    patch_lines = (len(prevlines) + 1, len(prevlines) + len(lines))
    return "\n".join(merged_lines), patch_lines


class EvidenceRetriever:
    def __init__(self, max_lines_per_ref: int = 3):
        self.max_lines_per_ref = max_lines_per_ref

    def _candidate_lines(self, patch: str, old_file: str) -> List[str]:
        candidates: List[str] = []
        meta_prefixes = ("diff --git", "index ", "--- ", "+++ ", "@@ ")
        for ln in patch.splitlines():
            if not ln or ln.startswith(meta_prefixes):
                continue
            if ln[0] in {"+", "-", " "} and ln[1:].strip():
                candidates.append(ln[1:].strip())
        new_file, span = ("", (-1, -1))
        try:
            new_file, span = generate_newf(old_file, patch)
        except Exception:
            pass
        if new_file and span != (-1, -1):
            nf_lines = new_file.splitlines()
            start = max(span[0] - 3, 0)
            end = min(span[1] + 2, len(nf_lines))
            for ln in nf_lines[start:end]:
                ln = ln.strip()
                if ln and ln not in candidates:
                    candidates.append(ln)
        return candidates

    def retrieve(self, pseudo_refs: List[str], patch: str, old_file: str) -> Dict[str, List[str]]:
        candidates = self._candidate_lines(patch, old_file)
        evidence: Dict[str, List[str]] = {}
        for pref in pseudo_refs:
            scored = sorted(candidates, key=lambda ln: lexical_overlap(pref, ln), reverse=True)
            top = [ln for ln in scored if ln.strip()][: self.max_lines_per_ref]
            if not top and candidates:
                top = candidates[: self.max_lines_per_ref]
            evidence[pref] = top
        return evidence
