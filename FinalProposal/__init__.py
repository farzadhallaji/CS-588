"""
FinalProposal: SoftCRScore + evidence-grounded preference learning pipeline.

All code lives in this folder to avoid touching the existing ProposedApproach code.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow importing ProposedApproach utilities (data loading, scoring, editors).
ROOT = Path(__file__).resolve().parent
PROPOSED_ROOT = ROOT.parent / "ProposedApproach"
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.append(str(PROPOSED_ROOT))

__all__ = ["ROOT", "PROPOSED_ROOT"]
