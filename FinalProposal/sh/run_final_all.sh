#!/usr/bin/env bash
set -euo pipefail

# FinalProposal end-to-end runner:
# 1) generate candidates
# 2) select best via SoftCRScore + evidence grounding
# 3) evaluate with CRScore (same tau passed through)
# 4) build preference pairs + robustness + human export

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATA="${RAW_DATA:-$ROOT/../CRScore/human_study/phase1/raw_data.json}"
TAU="${TAU:-0.7314}"
MODEL_PATH="${MODEL_PATH:-mixedbread-ai/mxbai-embed-large-v1}"
MODEL_TYPE="${MODEL_TYPE:-ollama}" # ollama|hf-local|echo|lora
MODEL_NAME="${MODEL_NAME:-llama3:8b-instruct-q4_0}"
PROMPTS="${PROMPTS:-default,evidence_grounded,test_heavy,concise}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
CAND_FILE="${CAND_FILE:-$OUT_DIR/candidates.jsonl}"
SELECT_FILE="${SELECT_FILE:-$OUT_DIR/selected.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$OUT_DIR/selected_summary.json}"
PREF_FILE="${PREF_FILE:-$OUT_DIR/preferences.jsonl}"
ROBUST_FILE="${ROBUST_FILE:-$OUT_DIR/robustness.csv}"
HUMAN_EXPORT="${HUMAN_EXPORT:-$OUT_DIR/human_export.csv}"
SYSTEM_B="${SYSTEM_B:-}" # set to another outputs JSONL to create overlapped human export

mkdir -p "$OUT_DIR"

echo "=== Generate candidates (model_type=$MODEL_TYPE, model=$MODEL_NAME) ==="
python "$ROOT/generate_candidates.py" \
  --raw-data "$RAW_DATA" \
  --split test \
  --tau "$TAU" \
  --model-type "$MODEL_TYPE" \
  --model-name "$MODEL_NAME" \
  --prompt-variants "$PROMPTS" \
  --output "$CAND_FILE"

echo "=== Select best via SoftCRScore + evidence ==="
python "$ROOT/select_best.py" \
  --candidates "$CAND_FILE" \
  --tau "$TAU" \
  --model-path "$MODEL_PATH" \
  --output "$SELECT_FILE"

echo "=== Evaluate with CRScore (tau propagated) ==="
python "$ROOT/../ProposedApproach/evaluate.py" \
  --raw-data "$RAW_DATA" \
  --split test \
  --tau "$TAU" \
  --model-path "$MODEL_PATH" \
  --outputs "$SELECT_FILE" \
  --summary-out "$SUMMARY_FILE"

echo "=== Build DPO preference pairs ==="
python "$ROOT/build_preferences.py" \
  --candidates "$CAND_FILE" \
  --tau "$TAU" \
  --model-path "$MODEL_PATH" \
  --output "$PREF_FILE" \
  --include-median

echo "=== Robustness stress tests ==="
python "$ROOT/robustness_suite.py" \
  --outputs "$SELECT_FILE" \
  --raw-data "$RAW_DATA" \
  --tau "$TAU" \
  --output-csv "$ROBUST_FILE"

if [[ -n "$SYSTEM_B" && -s "$SYSTEM_B" ]]; then
  echo "=== Export paired human study CSV (overlap ensured) ==="
  python "$ROOT/export_human_study.py" \
    --system-a "$SELECT_FILE" \
    --system-b "$SYSTEM_B" \
    --raw-data "$RAW_DATA" \
    --output "$HUMAN_EXPORT"
else
  echo "Skipping human export; provide SYSTEM_B to compare against another system output."
fi

echo "All FinalProposal runs finished. Outputs: $OUT_DIR"
