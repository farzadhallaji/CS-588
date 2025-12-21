"""
Compute CRScore across baseline/model columns in enhanced_code_reviews.csv and
external system outputs (threshold-gated DeepSeek default/concise, rewrite_loop).

Run:
  python scripts/compute_combined_crscore.py

Outputs:
  - results/combined_crscores.json: per-system overall and per-language Con/Comp/Rel,
    plus deltas vs baseline (msg) and enhanced_comment.
  - results/combined_crscores.csv: flat table for quick inspection.
  - analysis/plots/combined_rel.png: bar chart of overall Rel by system.
  - analysis/plots/combined_rel_by_lang.png: grouped bar chart of Rel by system/lang.
  - Console summary with top systems.
"""

from __future__ import annotations

import csv
import json
import sys
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scoring import CRScorer  # noqa: E402

# Hard-coded inputs/outputs
CSV_PATH = ROOT / "enhanced_code_reviews.csv"
THRESHOLD_DEFAULT_PATH = ROOT / "results" / "threshold_default_deepseek-coder_6.7b-base-q4_0.jsonl"
THRESHOLD_CONCISE_PATH = ROOT / "results" / "threshold_concise_deepseek-coder_6.7b-base-q4_0.jsonl"
REWRITE_LOOP_PATH = ROOT / "results" / "rewrite_loop.jsonl"
MODEL_PATH = "mixedbread-ai/mxbai-embed-large-v1"
TAU = 0.7314
OUTPUT_JSON = ROOT / "results" / "combined_crscores.json"
OUTPUT_CSV = ROOT / "results" / "combined_crscores.csv"
PLOT_DIR = ROOT / "analysis" / "plots"
BASELINE_COL = "msg"
ENHANCED_COL = "enhanced_comment"

# Columns from the CSV to score
CSV_COLUMNS = [
    "msg",
    "codereviewer_pred",
    "magicoder_pred",
    "deepseekcoder_pred",
    "stable_code_pred",
    "llama3_pred",
    "gpt3.5_pred",
    "codellama_7b_pred",
    "codellama_7b_pred.1",
    "lstm_pred",
    "knn_pred",
    "enhanced_comment",
]


def parse_claims(raw: str) -> List[str]:
    if not raw:
        return []
    try:
        obj = literal_eval(raw)
    except Exception:
        return []
    claims: List[str] = []
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                text = item[1]
            elif isinstance(item, str):
                text = item
            else:
                continue
            if text:
                claims.append(str(text))
    elif isinstance(obj, str):
        claims.append(obj)
    return claims


def aggregate(scores: Sequence[float]) -> float:
    return float(mean(scores)) if scores else 0.0


def load_jsonl_reviews(path: Path, text_keys: Iterable[str]) -> Dict[int, str]:
    idx_to_review: Dict[int, str] = {}
    if not path.exists():
        return idx_to_review
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("instance", {}).get("meta", {})
            idx_raw = meta.get("index")
            if idx_raw is None:
                continue
            try:
                idx = int(idx_raw)
            except Exception:
                continue
            text = None
            for key in text_keys:
                if rec.get(key):
                    text = rec[key]
                    break
            if text:
                idx_to_review[idx] = text
    return idx_to_review


@dataclass
class MetricBundle:
    con: float
    comp: float
    rel: float
    count: int


def score_systems():
    rows = list(csv.DictReader(CSV_PATH.open()))
    claims_map = {int(row["index"]): parse_claims(row.get("claims", "")) for row in rows}
    lang_map = {int(row["index"]): row["lang"] for row in rows}

    # External systems keyed by index -> review text
    threshold_default = load_jsonl_reviews(THRESHOLD_DEFAULT_PATH, ["best_review", "refined_review", "review"])
    threshold_concise = load_jsonl_reviews(THRESHOLD_CONCISE_PATH, ["best_review", "refined_review", "review"])
    rewrite_loop = load_jsonl_reviews(REWRITE_LOOP_PATH, ["best_review", "review"])

    # Collect reviews per system
    reviews: Dict[str, List[Tuple[List[str], str, str]]] = {col: [] for col in CSV_COLUMNS}
    reviews.update(
        {
            "threshold_default_deepseek": [],
            "threshold_concise_deepseek": [],
            "rewrite_loop": [],
        }
    )

    for row in rows:
        idx = int(row["index"])
        claims = claims_map.get(idx, [])
        lang = lang_map.get(idx, "unknown")
        for col in CSV_COLUMNS:
            text = (row.get(col) or "").strip()
            if text:
                reviews[col].append((claims, text, lang))
        if idx in threshold_default:
            reviews["threshold_default_deepseek"].append((claims, threshold_default[idx], lang))
        if idx in threshold_concise:
            reviews["threshold_concise_deepseek"].append((claims, threshold_concise[idx], lang))
        if idx in rewrite_loop:
            reviews["rewrite_loop"].append((claims, rewrite_loop[idx], lang))

    scorer = CRScorer(model_path=MODEL_PATH, tau=TAU)

    summary = {}
    for name, triples in reviews.items():
        cons: List[float] = []
        comps: List[float] = []
        rels: List[float] = []
        per_lang: Dict[str, Dict[str, List[float]]] = {}
        for claims, review, lang in triples:
            result = scorer.score(claims, review)
            cons.append(result.conciseness)
            comps.append(result.comprehensiveness)
            rels.append(result.relevance)
            bucket = per_lang.setdefault(lang, {"Con": [], "Comp": [], "Rel": []})
            bucket["Con"].append(result.conciseness)
            bucket["Comp"].append(result.comprehensiveness)
            bucket["Rel"].append(result.relevance)
        overall = {
            "count_scored": len(triples),
            "Con": aggregate(cons),
            "Comp": aggregate(comps),
            "Rel": aggregate(rels),
        }
        by_lang = {
            lang: {
                "count_scored": len(vals["Rel"]),
                "Con": aggregate(vals["Con"]),
                "Comp": aggregate(vals["Comp"]),
                "Rel": aggregate(vals["Rel"]),
            }
            for lang, vals in per_lang.items()
        }
        summary[name] = {"overall": overall, "by_lang": by_lang}

    baseline_rel = summary.get(BASELINE_COL, {}).get("overall", {}).get("Rel", 0.0)
    enhanced_rel = summary.get(ENHANCED_COL, {}).get("overall", {}).get("Rel", 0.0)
    for metrics in summary.values():
        rel = metrics["overall"]["Rel"]
        metrics["overall"]["delta_rel_vs_baseline"] = rel - baseline_rel
        metrics["overall"]["delta_rel_vs_enhanced"] = rel - enhanced_rel

    return summary


def save_summary(summary: Dict[str, Dict]):
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))

    # Flat CSV for quick inspection
    rows = []
    for sys_name, metrics in summary.items():
        o = metrics["overall"]
        rows.append(
            {
                "system": sys_name,
                "count": o["count_scored"],
                "Con": o["Con"],
                "Comp": o["Comp"],
                "Rel": o["Rel"],
                "delta_rel_vs_baseline": o.get("delta_rel_vs_baseline", 0.0),
                "delta_rel_vs_enhanced": o.get("delta_rel_vs_enhanced", 0.0),
            }
        )
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "count", "Con", "Comp", "Rel", "delta_rel_vs_baseline", "delta_rel_vs_enhanced"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: Dict[str, Dict]):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Overall Rel bar chart
    systems = []
    rels = []
    for name, metrics in sorted(summary.items(), key=lambda kv: kv[1]["overall"]["Rel"], reverse=True):
        systems.append(name)
        rels.append(metrics["overall"]["Rel"])
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(systems)), rels, color="steelblue")
    plt.xticks(range(len(systems)), systems, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Rel")
    plt.title("CRScore Rel by system")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "combined_rel.png", dpi=200)
    plt.close()

    # Per-language grouped bars (Rel)
    langs = ["java", "js", "py"]
    x = range(len(systems))
    width = 0.25
    offsets = [-width, 0, width]
    plt.figure(figsize=(10, 5))
    for off, lang, color in zip(offsets, langs, ["tomato", "seagreen", "royalblue"]):
        rel_vals = []
        for name in systems:
            rel_vals.append(summary[name]["by_lang"].get(lang, {}).get("Rel", 0.0))
        plt.bar([xi + off for xi in x], rel_vals, width=width, label=lang, color=color)
    plt.xticks(range(len(systems)), systems, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Rel by language")
    plt.title("CRScore Rel by system and language")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "combined_rel_by_lang.png", dpi=200)
    plt.close()


def print_console_summary(summary: Dict[str, Dict]):
    print("=== Overall (sorted by Rel) ===")
    sorted_items = sorted(summary.items(), key=lambda kv: kv[1]["overall"]["Rel"], reverse=True)
    for name, metrics in sorted_items:
        o = metrics["overall"]
        print(
            f"{name:30s} Rel={o['Rel']:.3f} Con={o['Con']:.3f} Comp={o['Comp']:.3f} "
            f"dRel_vs_msg={o.get('delta_rel_vs_baseline', 0.0):+.3f} "
            f"dRel_vs_enh={o.get('delta_rel_vs_enhanced', 0.0):+.3f} n={o['count_scored']}"
        )
    print("\nTop 5 by Rel:")
    for name, metrics in sorted_items[:5]:
        o = metrics["overall"]
        print(f"- {name}: Rel {o['Rel']:.3f} (Con {o['Con']:.3f}, Comp {o['Comp']:.3f})")


def main():
    summary = score_systems()
    save_summary(summary)
    plot_summary(summary)
    print_console_summary(summary)
    print(f"\nSaved JSON to {OUTPUT_JSON}")
    print(f"Saved CSV to  {OUTPUT_CSV}")
    if PLOT_DIR.exists():
        print(f"Plots in     {PLOT_DIR}")


if __name__ == "__main__":
    main()
