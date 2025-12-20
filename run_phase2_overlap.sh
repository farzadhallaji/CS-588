#!/usr/bin/env bash
set -euo pipefail

# Phase2 overlap evaluation for top systems.
# Requires: sentence-transformers, torch, and existing outputs JSONL for the systems below.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DATA="${RAW_DATA:-$ROOT/../CRScore/human_study/phase1/raw_data.json}"
PHASE2_DIR="${PHASE2_DIR:-$ROOT/../CRScore/human_study/phase2}"
PHASE2_PATTERN="${PHASE2_PATTERN:-*review_qual*_final.csv}"
BASELINE_SYS="${BASELINE_SYS:-msg}"
TARGET_SYS="${TARGET_SYS:-gpt3.5_pred}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"

ensure_pydeps() {
  python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("sentence_transformers")
importlib.import_module("torch")
PY
}

run_eval() {
  local outputs_file="$1"
  local summary_out="$2"
  echo "=== Phase2 overlap for $(basename "$outputs_file") ==="
  python "$ROOT/scripts/phase2_overlap_eval.py" \
    --raw-data "$RAW_DATA" \
    --phase2-dir "$PHASE2_DIR" \
    --phase2-pattern "$PHASE2_PATTERN" \
    --baseline-system "$BASELINE_SYS" \
    --target-system "$TARGET_SYS" \
    --outputs "$outputs_file" \
    --summary-out "$summary_out"
}

if ! ensure_pydeps; then
  echo "Missing deps (sentence-transformers/torch). Install them before running."
  exit 1
fi

# Threshold best systems
run_eval "$OUT_DIR/threshold_default_deepseek-coder_6.7b-base-q4_0.jsonl" "$OUT_DIR/phase2_overlap_default_deepseek.json"
run_eval "$OUT_DIR/threshold_concise_deepseek-coder_6.7b-base-q4_0.jsonl" "$OUT_DIR/phase2_overlap_concise_deepseek.json"
run_eval "$OUT_DIR/threshold_default_llama3_8b-instruct-q4_0.jsonl" "$OUT_DIR/phase2_overlap_default_llama3.json"

# Offline baseline
run_eval "$OUT_DIR/rewrite_loop.jsonl" "$OUT_DIR/phase2_overlap_rewrite_loop.json"

# Proposal v1 (best models)
run_eval "$OUT_DIR/proposal_v1_qwen2.5-coder_7b.jsonl" "$OUT_DIR/phase2_overlap_proposal_qwen.json"
run_eval "$OUT_DIR/proposal_v1_llama3_8b-instruct-q4_0.jsonl" "$OUT_DIR/phase2_overlap_proposal_llama3.json"

echo "Phase2 overlap evaluations complete."
