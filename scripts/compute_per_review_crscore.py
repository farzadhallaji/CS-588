"""
Compute CRScore for every available review per instance (human msg, enhanced review, model outputs).

Run:
  python scripts/compute_per_review_crscore.py

Outputs:
  - results/per_review_crscores_full.csv: index/id/lang/system/review text with Con/Comp/Rel scores.
  - results/per_row_crscores.csv: same scores without review text.
  - results/combined_crscores.json|.csv: aggregated metrics per system (overall + by language, with deltas).
  - analysis/plots/combined_rel.png / combined_rel_by_lang.png: bar charts of Rel metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scoring import CRScorer  # noqa: E402

RAW_DATA_DEFAULT = ROOT.parent / "CRScore" / "human_study" / "phase1" / "raw_data.json"
ENHANCED_CSV_DEFAULT = ROOT / "enhanced_code_reviews_with_index.csv"
THRESHOLD_DEFAULT_PATH = ROOT / "results" / "threshold_default_deepseek-coder_6.7b-base-q4_0.jsonl"
THRESHOLD_CONCISE_PATH = ROOT / "results" / "threshold_concise_deepseek-coder_6.7b-base-q4_0.jsonl"
REWRITE_LOOP_PATH = ROOT / "results" / "rewrite_loop.jsonl"
OUTPUT_PATH_DEFAULT = ROOT / "results" / "per_review_crscores_full.csv"
PER_ROW_OUTPUT = ROOT / "results" / "per_row_crscores.csv"
OUTPUT_JSON = ROOT / "results" / "combined_crscores.json"
OUTPUT_SUMMARY_CSV = ROOT / "results" / "combined_crscores.csv"
PLOT_DIR = ROOT / "analysis" / "plots"
MODEL_PATH_DEFAULT = "mixedbread-ai/mxbai-embed-large-v1"
TAU_DEFAULT = 0.7314
BASELINE_COL = "msg"
ENHANCED_COL = "enhanced_comment"

# Reviews stored directly in raw_data.json
RAW_REVIEW_FIELDS: Sequence[str] = [
    "msg",
    "codereviewer_pred",
    "magicoder_pred",
    "deepseekcoder_pred",
    "stable_code_pred",
    "llama3_pred",
    "gpt3.5_pred",
    "codellama_7b_pred",
    "codellama_13b_pred",
    "lstm_pred",
    "knn_pred",
]


def parse_claims(obj: Iterable) -> List[str]:
    """Convert the claims object (list/tuple/str) to a flat list of strings."""
    claims: List[str] = []
    if not obj:
        return claims
    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            claims.append(str(item[1]))
        elif isinstance(item, str):
            claims.append(item)
    return claims


def load_enhanced_comments(path: Path) -> Dict[int, str]:
    """Map index -> enhanced_comment (first occurrence wins)."""
    mapping: Dict[int, str] = {}
    if not path.exists():
        return mapping
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["index"])
            except Exception:
                continue
            if idx in mapping:
                continue
            text = (row.get("enhanced_comment") or "").strip()
            if text:
                mapping[idx] = text
    return mapping


def load_jsonl_reviews(path: Path, text_keys: Iterable[str]) -> Dict[int, str]:
    """Load external system outputs keyed by instance index."""
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


def collect_reviews(raw_path: Path, enhanced_path: Path) -> Dict[int, Dict[str, str]]:
    """Gather all review texts per index."""
    records = json.loads(raw_path.read_text())
    enhanced = load_enhanced_comments(enhanced_path)
    threshold_default = load_jsonl_reviews(THRESHOLD_DEFAULT_PATH, ["best_review", "refined_review", "review"])
    threshold_concise = load_jsonl_reviews(THRESHOLD_CONCISE_PATH, ["best_review", "refined_review", "review"])
    rewrite_loop = load_jsonl_reviews(REWRITE_LOOP_PATH, ["best_review", "review"])

    idx_to_reviews: Dict[int, Dict[str, str]] = {}
    for rec in records:
        try:
            idx = int(rec["index"])
        except Exception:
            continue
        reviews: Dict[str, str] = {}
        for field in RAW_REVIEW_FIELDS:
            text = (rec.get(field) or "").strip()
            if text:
                reviews[field] = text
        if idx in enhanced:
            reviews["enhanced_comment"] = enhanced[idx]
        if idx in threshold_default:
            reviews["threshold_default_deepseek"] = threshold_default[idx]
        if idx in threshold_concise:
            reviews["threshold_concise_deepseek"] = threshold_concise[idx]
        if idx in rewrite_loop:
            reviews["rewrite_loop"] = rewrite_loop[idx]
        if reviews:
            idx_to_reviews[idx] = reviews
    return idx_to_reviews


def aggregate(scores: Sequence[float]) -> float:
    return float(mean(scores)) if scores else 0.0


def summarize(per_row: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Aggregate overall/by-lang metrics and compute deltas."""
    summary: Dict[str, Dict[str, Any]] = {}
    for row in per_row:
        system = row["system"]
        lang = row["lang"]
        bucket = summary.setdefault(system, {"overall": {"Con": [], "Comp": [], "Rel": []}, "by_lang": {}})
        bucket["overall"]["Con"].append(row["Con"])
        bucket["overall"]["Comp"].append(row["Comp"])
        bucket["overall"]["Rel"].append(row["Rel"])
        lang_bucket = bucket["by_lang"].setdefault(lang, {"Con": [], "Comp": [], "Rel": []})
        lang_bucket["Con"].append(row["Con"])
        lang_bucket["Comp"].append(row["Comp"])
        lang_bucket["Rel"].append(row["Rel"])

    for sys_name, metrics in summary.items():
        o = metrics["overall"]
        metrics["overall"] = {
            "count_scored": len(o["Rel"]),
            "Con": aggregate(o["Con"]),
            "Comp": aggregate(o["Comp"]),
            "Rel": aggregate(o["Rel"]),
        }
        by_lang = {}
        for lang, vals in metrics["by_lang"].items():
            by_lang[lang] = {
                "count_scored": len(vals["Rel"]),
                "Con": aggregate(vals["Con"]),
                "Comp": aggregate(vals["Comp"]),
                "Rel": aggregate(vals["Rel"]),
            }
        metrics["by_lang"] = by_lang

    baseline_rel = summary.get(BASELINE_COL, {}).get("overall", {}).get("Rel", 0.0)
    enhanced_rel = summary.get(ENHANCED_COL, {}).get("overall", {}).get("Rel", 0.0)
    for metrics in summary.values():
        rel = metrics["overall"]["Rel"]
        metrics["overall"]["delta_rel_vs_baseline"] = rel - baseline_rel
        metrics["overall"]["delta_rel_vs_enhanced"] = rel - enhanced_rel
    return summary


def score_reviews(
    idx_to_reviews: Dict[int, Dict[str, str]],
    raw_path: Path,
    model_path: str,
    tau: float,
) -> List[Dict[str, Any]]:
    raw_records = {int(rec["index"]): rec for rec in json.loads(raw_path.read_text())}
    scorer = CRScorer(model_path=model_path, tau=tau)

    rows: List[Dict[str, Any]] = []
    for idx, reviews in idx_to_reviews.items():
        rec = raw_records.get(idx, {})
        claims = parse_claims(rec.get("claims", []))
        lang = rec.get("lang", "unknown")
        rid = rec.get("id", "")
        for system, text in reviews.items():
            result = scorer.score(claims, text)
            rows.append(
                {
                    "index": idx,
                    "id": rid,
                    "lang": lang,
                    "system": system,
                    "review": text,
                    "Con": result.conciseness,
                    "Comp": result.comprehensiveness,
                    "Rel": result.relevance,
                }
            )
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "id", "lang", "system", "review", "Con", "Comp", "Rel"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "Con": f"{row['Con']:.6f}",
                    "Comp": f"{row['Comp']:.6f}",
                    "Rel": f"{row['Rel']:.6f}",
                }
            )


def write_per_row_no_text(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "id", "lang", "system", "Con", "Comp", "Rel"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row["index"],
                    "id": row["id"],
                    "lang": row["lang"],
                    "system": row["system"],
                    "Con": f"{row['Con']:.6f}",
                    "Comp": f"{row['Comp']:.6f}",
                    "Rel": f"{row['Rel']:.6f}",
                }
            )


def save_summary(summary: Dict[str, Dict]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))

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
    with OUTPUT_SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "count", "Con", "Comp", "Rel", "delta_rel_vs_baseline", "delta_rel_vs_enhanced"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: Dict[str, Dict]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

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


def print_console_summary(summary: Dict[str, Dict]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CRScore per review for each instance.")
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA_DEFAULT, help="Path to phase1 raw_data.json.")
    parser.add_argument(
        "--enhanced-csv",
        type=Path,
        default=ENHANCED_CSV_DEFAULT,
        help="CSV containing enhanced_comment to merge (optional).",
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH_DEFAULT, help="Embedding model path.")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT, help="CRScore similarity threshold.")
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_PATH_DEFAULT, help="Where to write the scored CSV (with review text)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    idx_to_reviews = collect_reviews(args.raw_data, args.enhanced_csv)
    rows = score_reviews(idx_to_reviews, args.raw_data, args.model_path, args.tau)
    write_csv(rows, args.out)
    write_per_row_no_text(rows, PER_ROW_OUTPUT)
    summary = summarize(rows)
    save_summary(summary)
    plot_summary(summary)
    print_console_summary(summary)
    print(f"\nWrote {len(rows)} scored reviews to {args.out}")
    print(f"Wrote per-row scores (no text) to {PER_ROW_OUTPUT}")
    print(f"Summary JSON to {OUTPUT_JSON}")
    print(f"Summary CSV  to {OUTPUT_SUMMARY_CSV}")
    if PLOT_DIR.exists():
        print(f"Plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
