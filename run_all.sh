#!/usr/bin/env bash
set -euo pipefail

# Single entry point: runs main loop, ablations, and threshold-gated local LLM refinements.
# Configure with env vars as needed:
#   MODEL_PATH=... (SentenceTransformer for scoring)
#   RAW_DATA=...   (path to raw_data.json)
#   TAU=..., TAU_EVIDENCE=..., MAX_CHANGE=...
#   OLLAMA_MODEL=... (for local LLM editing; default llama3:8b-instruct-q4_0)
#   QWEN_MODEL_PATH=... (optional HF local model for threshold refinement)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DATA="${RAW_DATA:-$ROOT/../CRScore/human_study/phase1/raw_data.json}"
MODEL_PATH="${MODEL_PATH:-mixedbread-ai/mxbai-embed-large-v1}"
TAU="${TAU:-0.7314}"
TAU_EVIDENCE="${TAU_EVIDENCE:-0.35}"
MAX_CHANGE="${MAX_CHANGE:-0.7}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3:8b-instruct-q4_0}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
HF_OUT_DIR="${HF_OUT_DIR:-$ROOT/results_hf}"
PHASE2_DIR="${PHASE2_DIR:-$ROOT/../CRScore/human_study/phase2}"
SPLITS_OUT="${SPLITS_OUT:-$OUT_DIR/splits.json}"
FEWSHOT_OUT="${FEWSHOT_OUT:-$OUT_DIR/fewshot_pairs.json}"
PROPOSAL_OUT="${PROPOSAL_OUT:-$OUT_DIR/proposal_v1.jsonl}"
PROPOSAL_SUMMARY="${PROPOSAL_SUMMARY:-$OUT_DIR/proposal_v1_summary.json}"
HUMAN_BASELINE_SYSTEM="${HUMAN_BASELINE_SYSTEM:-msg}"
HUMAN_TARGET_SYSTEM="${HUMAN_TARGET_SYSTEM:-gpt3.5_pred}"
HUMAN_CORR_SYSTEM="${HUMAN_CORR_SYSTEM:-}"
HUMAN_OUTPUTS="${HUMAN_OUTPUTS:-}"
HUMAN_SUMMARY_OUT="${HUMAN_SUMMARY_OUT:-$OUT_DIR/human_eval_summary.json}"
EXTRA_OLLAMA_MODELS="${EXTRA_OLLAMA_MODELS:-deepseek-coder:6.7b-base-q4_0,qwen2.5-coder:3b,qwen2.5-coder:7b}" # comma-separated
THRESHOLD_PROMPTS="${THRESHOLD_PROMPTS:-default,concise,evidence,test-heavy}"

mkdir -p "$OUT_DIR"

ensure_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not installed; skipping ollama-backed runs."
    return 1
  fi
  if ! curl -sSf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Starting ollama daemon..."
    nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
    sleep 5
    if ! curl -sSf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      echo "Failed to reach ollama daemon; skipping ollama-backed runs."
      return 1
    fi
  fi
  return 0
}

have_model() {
  local model="$1"
  curl -s http://127.0.0.1:11434/api/tags | grep -q "\"name\":\"${model}\""
}

run_or_skip() {
  local target="$1"
  shift
  if [[ -s "$target" ]]; then
    echo "Skipping; found $target"
    return 0
  fi
  echo "Running: $*"
  "$@"
}

ensure_model() {
  local model="$1"
  if have_model "$model"; then
    return 0
  fi
  echo "Pulling ollama model '$model'..."
  if ollama pull "$model"; then
    return 0
  fi
  echo "Failed to pull '$model'; skipping this model."
  return 1
}

slug() {
  echo "$1" | tr '/:' '__' | tr ' ' '_' | tr '[:upper:]' '[:lower:]'
}

echo "=== Freeze deterministic dev/test split ==="
run_or_skip "$SPLITS_OUT" python "$ROOT/scripts/make_splits.py" --raw-data "$RAW_DATA" --out "$SPLITS_OUT"

echo "=== Build few-shot pairs for proposal v1 baseline ==="
run_or_skip "$FEWSHOT_OUT" python "$ROOT/scripts/build_fewshot_pairs.py" --raw-data "$RAW_DATA" --tau "$TAU" --model-path "$MODEL_PATH" --out "$FEWSHOT_OUT"

echo "=== Baseline (human seed) ==="
run_or_skip "$OUT_DIR/baseline_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --baseline-only --summary-out "$OUT_DIR/baseline_summary.json"

echo "=== Main loop + ablations (template editor) ==="
run_or_skip "$OUT_DIR/loop.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode loop --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/loop.jsonl"
run_or_skip "$OUT_DIR/single_edit.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode k1 --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/single_edit.jsonl"
run_or_skip "$OUT_DIR/single_rewrite.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode rewrite --max-iter 1 --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change 1.0 --output "$OUT_DIR/single_rewrite.jsonl"
run_or_skip "$OUT_DIR/no_selection.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode no-selection --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/no_selection.jsonl"
run_or_skip "$OUT_DIR/no_evidence.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode no-evidence --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/no_evidence.jsonl"
run_or_skip "$OUT_DIR/rewrite_loop.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode rewrite --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change 1.0 --output "$OUT_DIR/rewrite_loop.jsonl"

echo "=== Evaluate loop outputs ==="
run_or_skip "$OUT_DIR/loop_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/loop.jsonl" --summary-out "$OUT_DIR/loop_summary.json"
run_or_skip "$OUT_DIR/single_edit_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/single_edit.jsonl" --summary-out "$OUT_DIR/single_edit_summary.json"
run_or_skip "$OUT_DIR/single_rewrite_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/single_rewrite.jsonl" --summary-out "$OUT_DIR/single_rewrite_summary.json"
run_or_skip "$OUT_DIR/no_selection_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/no_selection.jsonl" --summary-out "$OUT_DIR/no_selection_summary.json"
run_or_skip "$OUT_DIR/no_evidence_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/no_evidence.jsonl" --summary-out "$OUT_DIR/no_evidence_summary.json"
run_or_skip "$OUT_DIR/rewrite_loop_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/rewrite_loop.jsonl" --summary-out "$OUT_DIR/rewrite_loop_summary.json"

echo "=== Proposal v1 few-shot baseline (Ollama sweep) ==="
if ensure_ollama; then
  MODELS_LIST="$OLLAMA_MODEL"
  if [[ -n "$EXTRA_OLLAMA_MODELS" ]]; then
    MODELS_LIST="$MODELS_LIST,$EXTRA_OLLAMA_MODELS"
  fi
  IFS=',' read -r -a MODELS <<<"$MODELS_LIST"
  for MODEL in "${MODELS[@]}"; do
    MODEL_TRIMMED="$(echo "$MODEL" | xargs)"
    [[ -z "$MODEL_TRIMMED" ]] && continue
    if ! ensure_model "$MODEL_TRIMMED"; then
      continue
    fi
    MODEL_SLUG=$(slug "$MODEL_TRIMMED")
    OUT_FILE="$OUT_DIR/proposal_v1_${MODEL_SLUG}.jsonl"
    SUM_FILE="$OUT_DIR/proposal_v1_${MODEL_SLUG}_summary.json"
    run_or_skip "$OUT_FILE" python "$ROOT/proposal_v1.py" \
      --raw-data "$RAW_DATA" \
      --split test \
      --tau "$TAU" \
      --model-path "$MODEL_PATH" \
      --fewshot "$FEWSHOT_OUT" \
      --ollama-model "$MODEL_TRIMMED" \
      --out "$OUT_FILE"
    run_or_skip "$SUM_FILE" python "$ROOT/evaluate.py" \
      --raw-data "$RAW_DATA" \
      --split test \
      --tau "$TAU" --model-path "$MODEL_PATH" \
      --outputs "$OUT_FILE" \
      --summary-out "$SUM_FILE"
  done
else
  echo "Skipping proposal v1 baseline; ollama not available."
fi

echo "=== Threshold-gated refinement (prompt/model sweep) ==="
if ensure_ollama; then
  MODELS_LIST="$OLLAMA_MODEL"
  if [[ -n "$EXTRA_OLLAMA_MODELS" ]]; then
    MODELS_LIST="$MODELS_LIST,$EXTRA_OLLAMA_MODELS"
  fi
  IFS=',' read -r -a MODELS <<<"$MODELS_LIST"
  IFS=',' read -r -a PROMPTS <<<"$THRESHOLD_PROMPTS"
  for MODEL in "${MODELS[@]}"; do
    MODEL_TRIMMED="$(echo "$MODEL" | xargs)"
    [[ -z "$MODEL_TRIMMED" ]] && continue
    if ! ensure_model "$MODEL_TRIMMED"; then
      continue
    fi
    MODEL_SLUG=$(slug "$MODEL_TRIMMED")
    for PV in "${PROMPTS[@]}"; do
      PV_TRIMMED="$(echo "$PV" | xargs)"
      [[ -z "$PV_TRIMMED" ]] && continue
      OUT_FILE="$OUT_DIR/threshold_${PV_TRIMMED}_${MODEL_SLUG}.jsonl"
      SUM_FILE="$OUT_DIR/threshold_${PV_TRIMMED}_${MODEL_SLUG}_summary.json"
      run_or_skip "$OUT_FILE" python "$ROOT/threshold_refine.py" \
        --raw-data "$RAW_DATA" \
        --split test \
        --tau "$TAU" \
        --threshold 0.6 \
        --model-type ollama \
        --model-name "$MODEL_TRIMMED" \
        --prompt-variant "$PV_TRIMMED" \
        --output "$OUT_FILE"

      run_or_skip "$SUM_FILE" python "$ROOT/evaluate.py" \
        --raw-data "$RAW_DATA" \
        --split test \
        --tau "$TAU" --model-path "$MODEL_PATH" \
        --outputs "$OUT_FILE" \
        --summary-out "$SUM_FILE"
    done
  done
fi

if [[ -n "${QWEN_MODEL_PATH:-}" ]]; then
  echo "=== Threshold-gated refinement with HF local Qwen (evidence prompt) ==="
  mkdir -p "$HF_OUT_DIR"
  run_or_skip "$HF_OUT_DIR/threshold_qwen_evidence.jsonl" python "$ROOT/threshold_refine.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --threshold 0.6 \
    --model-type hf-local \
    --model-name "$QWEN_MODEL_PATH" \
    --prompt-variant evidence \
    --device "${QWEN_DEVICE:-cpu}" \
    --output "$HF_OUT_DIR/threshold_qwen_evidence.jsonl"

  run_or_skip "$HF_OUT_DIR/threshold_qwen_evidence_summary.json" python "$ROOT/evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" --model-path "$MODEL_PATH" \
    --outputs "$HF_OUT_DIR/threshold_qwen_evidence.jsonl" \
    --summary-out "$HF_OUT_DIR/threshold_qwen_evidence_summary.json"
else
  echo "QWEN_MODEL_PATH not set; skipping HF local Qwen threshold experiment."
fi

if [[ "${SKIP_HUMAN_EVAL:-0}" != "1" ]]; then
  echo "=== Phase2 human evaluation stats ==="
  if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("scipy.stats")
PY
  then
    HUMAN_ARGS=(--raw-data "$RAW_DATA" --phase2-dir "$PHASE2_DIR" --split test --tau "$TAU" --model-path "$MODEL_PATH" --baseline-system "$HUMAN_BASELINE_SYSTEM" --target-system "$HUMAN_TARGET_SYSTEM")
    if [[ -n "$HUMAN_CORR_SYSTEM" ]]; then
      HUMAN_ARGS+=(--corr-system "$HUMAN_CORR_SYSTEM")
    fi
    if [[ -n "$HUMAN_OUTPUTS" ]]; then
      HUMAN_ARGS+=(--outputs "$HUMAN_OUTPUTS")
    fi
    run_or_skip "$HUMAN_SUMMARY_OUT" python "$ROOT/human_eval.py" "${HUMAN_ARGS[@]}" --summary-out "$HUMAN_SUMMARY_OUT"
  else
    echo "Skipping human_eval.py because scipy is not installed."
  fi
fi

echo "All runs complete. Check summaries under $OUT_DIR and $HF_OUT_DIR."
