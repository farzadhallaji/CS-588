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
TAU="${TAU:-0.6}"
TAU_EVIDENCE="${TAU_EVIDENCE:-0.35}"
MAX_CHANGE="${MAX_CHANGE:-0.7}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3:8b-instruct-q4_0}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
HF_OUT_DIR="${HF_OUT_DIR:-$ROOT/results_hf}"

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

echo "=== Baseline (human seed) ==="
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --baseline-only --summary-out "$OUT_DIR/baseline_summary.json"

echo "=== Main loop + ablations (template editor) ==="
python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode loop --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/loop.jsonl"
python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode k1 --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/single_edit.jsonl"
python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode rewrite --max-iter 1 --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change 1.0 --output "$OUT_DIR/single_rewrite.jsonl"
python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode no-selection --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/no_selection.jsonl"
python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode no-evidence --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$OUT_DIR/no_evidence.jsonl"
python "$ROOT/run.py" --raw-data "$RAW_DATA" --split test --mode rewrite --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change 1.0 --output "$OUT_DIR/rewrite_loop.jsonl"

echo "=== Evaluate loop outputs ==="
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/loop.jsonl" --summary-out "$OUT_DIR/loop_summary.json"
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/single_edit.jsonl" --summary-out "$OUT_DIR/single_edit_summary.json"
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/single_rewrite.jsonl" --summary-out "$OUT_DIR/single_rewrite_summary.json"
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/no_selection.jsonl" --summary-out "$OUT_DIR/no_selection_summary.json"
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/no_evidence.jsonl" --summary-out "$OUT_DIR/no_evidence_summary.json"
python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split test --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$OUT_DIR/rewrite_loop.jsonl" --summary-out "$OUT_DIR/rewrite_loop_summary.json"

echo "=== Threshold-gated refinement with Ollama (${OLLAMA_MODEL}) ==="
if ensure_ollama; then
  python "$ROOT/threshold_refine.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --threshold 0.6 \
    --model-type ollama \
    --model-name "$OLLAMA_MODEL" \
    --prompt-variant default \
    --output "$OUT_DIR/threshold_ollama_default.jsonl"

  python "$ROOT/evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" --model-path "$MODEL_PATH" \
    --outputs "$OUT_DIR/threshold_ollama_default.jsonl" \
    --summary-out "$OUT_DIR/threshold_ollama_default_summary.json"

  echo "=== Threshold-gated refinement (concise prompt) with Ollama (${OLLAMA_MODEL}) ==="
  python "$ROOT/threshold_refine.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --threshold 0.6 \
    --model-type ollama \
    --model-name "$OLLAMA_MODEL" \
    --prompt-variant concise \
    --output "$OUT_DIR/threshold_ollama_concise.jsonl"

  python "$ROOT/evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" --model-path "$MODEL_PATH" \
    --outputs "$OUT_DIR/threshold_ollama_concise.jsonl" \
    --summary-out "$OUT_DIR/threshold_ollama_concise_summary.json"
fi

if [[ -n "${QWEN_MODEL_PATH:-}" ]]; then
  echo "=== Threshold-gated refinement with HF local Qwen (evidence prompt) ==="
  mkdir -p "$HF_OUT_DIR"
  python "$ROOT/threshold_refine.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --threshold 0.6 \
    --model-type hf-local \
    --model-name "$QWEN_MODEL_PATH" \
    --prompt-variant evidence \
    --device "${QWEN_DEVICE:-cpu}" \
    --output "$HF_OUT_DIR/threshold_qwen_evidence.jsonl"

  python "$ROOT/evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" --model-path "$MODEL_PATH" \
    --outputs "$HF_OUT_DIR/threshold_qwen_evidence.jsonl" \
    --summary-out "$HF_OUT_DIR/threshold_qwen_evidence_summary.json"
else
  echo "QWEN_MODEL_PATH not set; skipping HF local Qwen threshold experiment."
fi

echo "All runs complete. Check summaries under $OUT_DIR and $HF_OUT_DIR."
