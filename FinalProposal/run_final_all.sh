#!/usr/bin/env bash
set -euo pipefail

# FinalProposal end-to-end runner with ablations + resumability.
# Produces:
# - Reward rerank baseline (candidates -> select -> evaluate -> robustness)
# - Generation ablations (prompt variants, N samples, temperature)
# - Reward ablations (penalties on/off, tau/temp sweeps, hard CRScore)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DATA="${RAW_DATA:-$ROOT/../../CRScore/human_study/phase1/raw_data.json}"
TAU="${TAU:-0.7314}"
MODEL_PATH="${MODEL_PATH:-mixedbread-ai/mxbai-embed-large-v1}"
MODEL_TYPE="${MODEL_TYPE:-ollama}" # ollama|hf-local|echo|lora
MODEL_NAME="${MODEL_NAME:-llama3:8b-instruct-q4_0}"
PROMPTS="${PROMPTS:-default,evidence_grounded,test_heavy,concise}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
CAND_DIR="${CAND_DIR:-$OUT_DIR/candidates}"
SEL_DIR="${SEL_DIR:-$OUT_DIR/selected}"
SUM_DIR="${SUM_DIR:-$OUT_DIR/summaries}"
ROBUST_DIR="${ROBUST_DIR:-$OUT_DIR/robustness}"
SYSTEM_B="${SYSTEM_B:-}" # optional: export human overlap vs another system output
OLLAMA_STARTED=0

mkdir -p "$OUT_DIR" "$CAND_DIR" "$SEL_DIR" "$SUM_DIR" "$ROBUST_DIR"
export PYTHONPATH="$ROOT/..:${PYTHONPATH:-}"

ensure_ollama() {
  if [[ "$MODEL_TYPE" != "ollama" ]]; then
    return
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama CLI not found; install it or set MODEL_TYPE=hf-local/echo." >&2
    exit 1
  fi
  if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "Starting ollama serve in background (logs: /tmp/ollama-serve.log)..."
    nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
    sleep 3
    OLLAMA_STARTED=1
  else
    echo "ollama serve already running."
  fi
  if ! ollama list >/dev/null 2>&1; then
    echo "ollama serve did not become ready; check /tmp/ollama-serve.log" >&2
    exit 1
  fi
}

cleanup_ollama() {
  if [[ "$MODEL_TYPE" != "ollama" ]]; then
    return
  fi
  # Only stop the server if we started it in this script to free GPU memory.
  if [[ "$OLLAMA_STARTED" == "1" ]]; then
    echo "Stopping ollama serve to free GPU resources..."
    pkill -f "ollama serve" >/dev/null 2>&1 || true
  fi
}

trap cleanup_ollama EXIT

run_or_skip() {
  local target="$1"
  shift
  if [[ -s "$target" ]]; then
    echo "Skipping; found $target" >&2
    return 0
  fi
  echo "Running: $*" >&2
  "$@"
}

gen_candidates() {
  local cfg="$1"
  local name="" num_samples=2 temperature=0.3 prompts="$PROMPTS" limit=""
  for kv in $cfg; do
    local k="${kv%%=*}" v="${kv#*=}"
    eval "$k=\"$v\""
  done
  local cand_file="$CAND_DIR/candidates_${name}.jsonl"
  run_or_skip "$cand_file" python -m FinalProposal.generate_candidates \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --model-type "$MODEL_TYPE" \
    --model-name "$MODEL_NAME" \
    --prompt-variants "$prompts" \
    --num-samples "$num_samples" \
    --temperature "$temperature" \
    --output "$cand_file"
  echo "$cand_file"
}

select_and_eval() {
  local cand_file="$1"
  local cand_tag="$2"
  local cfg="$3"
  local name="" tau_val="$TAU" temp=0.05 evidence_margin=0.35 top_k_align=2 score_mode="soft"
  local w_rel=1.0 w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400
  for kv in $cfg; do
    local k="${kv%%=*}" v="${kv#*=}"
    eval "$k=\"$v\""
  done
  local select_file="$SEL_DIR/selected_${cand_tag}__${name}.jsonl"
  local summary_file="$SUM_DIR/summary_${cand_tag}__${name}.json"
  run_or_skip "$select_file" python -m FinalProposal.select_best \
    --candidates "$cand_file" \
    --tau "$tau_val" \
    --temp "$temp" \
    --evidence-margin "$evidence_margin" \
    --top-k-align "$top_k_align" \
    --model-path "$MODEL_PATH" \
    --score-mode "$score_mode" \
    --w-rel "$w_rel" \
    --w-unsupported "$w_unsupported" \
    --w-len "$w_len" \
    --w-copy "$w_copy" \
    --len-norm "$len_norm" \
    --output "$select_file"

  # Suppress stdout from evaluate to keep function return clean; summary is written to file.
  run_or_skip "$summary_file" python "$ROOT/../evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$tau_val" \
    --model-path "$MODEL_PATH" \
    --outputs "$select_file" \
    --summary-out "$summary_file" \
    >/dev/null

  echo "$select_file"
}

run_robustness() {
  local select_file="$1"
  local tag="$2"
  local tau_val="${3:-$TAU}"
  local rob_file="$ROBUST_DIR/robust_${tag}.csv"
  run_or_skip "$rob_file" python -m FinalProposal.robustness_suite \
    --outputs "$select_file" \
    --raw-data "$RAW_DATA" \
    --tau "$tau_val" \
    --model-path "$MODEL_PATH" \
    --output-csv "$rob_file"
}

# --------- Configurations ---------
BASE_GEN_CFG="name=base num_samples=2 temperature=0.3 prompts=${PROMPTS}"
GEN_ABLATIONS=(
  "name=temp02 num_samples=2 temperature=0.2 prompts=${PROMPTS}"
  "name=temp06 num_samples=2 temperature=0.6 prompts=${PROMPTS}"
  "name=num4 num_samples=4 temperature=0.3 prompts=${PROMPTS}"
  "name=num8 num_samples=8 temperature=0.3 prompts=${PROMPTS}"
  "name=prompt_default num_samples=2 temperature=0.3 prompts=default"
  "name=prompt_evidence num_samples=2 temperature=0.3 prompts=evidence_grounded"
  "name=prompt_concise num_samples=2 temperature=0.3 prompts=concise"
  "name=prompt_testheavy num_samples=2 temperature=0.3 prompts=test_heavy"
)

BASE_SCORE_CFG="name=reward_default score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU}"
SCORE_ABLATIONS=(
  "name=soft_only score_mode=soft w_unsupported=0 w_len=0 w_copy=0 len_norm=400 temp=0.05 tau=${TAU}"
  "name=soft_len score_mode=soft w_unsupported=0 w_len=0.02 w_copy=0 len_norm=400 temp=0.05 tau=${TAU}"
  "name=soft_evidence score_mode=soft w_unsupported=0.6 w_len=0 w_copy=0 len_norm=400 temp=0.05 tau=${TAU}"
  "name=soft_evidence_len score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0 len_norm=400 temp=0.05 tau=${TAU}"
  "name=soft_evidence_len_copy score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU}"
  "name=hard_cr score_mode=hard w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU}"
  "name=tau_low score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=0.65"
  "name=tau_high score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=0.80"
  "name=soft_temp_low score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.02 tau=${TAU}"
  "name=soft_temp_high score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.10 tau=${TAU}"
)

ensure_ollama

echo "=== Base FinalProposal reward rerank run ==="
BASE_CAND_FILE=$(gen_candidates "$BASE_GEN_CFG")
BASE_SELECT_FILE=$(select_and_eval "$BASE_CAND_FILE" "base" "$BASE_SCORE_CFG")
run_robustness "$BASE_SELECT_FILE" "base__reward_default" "$TAU"

echo "=== Generation ablations (default scoring) ==="
for cfg in "${GEN_ABLATIONS[@]}"; do
  gen_name=""
  for kv in $cfg; do
    k="${kv%%=*}"; v="${kv#*=}"; [[ "$k" == "name" ]] && gen_name="$v"
  done
  [[ -z "$gen_name" ]] && gen_name="gen"
  cand_file=$(gen_candidates "$cfg")
  select_and_eval "$cand_file" "gen_${gen_name}" "$BASE_SCORE_CFG"
done

echo "=== Reward/score ablations on base candidates ==="
for cfg in "${SCORE_ABLATIONS[@]}"; do
  score_name=""
  for kv in $cfg; do
    k="${kv%%=*}"; v="${kv#*=}"; [[ "$k" == "name" ]] && score_name="$v"
  done
  [[ -z "$score_name" ]] && score_name="score"
  select_and_eval "$BASE_CAND_FILE" "base" "$cfg"
done

if [[ -n "$SYSTEM_B" && -s "$SYSTEM_B" ]]; then
  echo "=== Export paired human study CSV (overlap ensured) ==="
  HUMAN_EXPORT="${HUMAN_EXPORT:-$OUT_DIR/human_export.csv}"
  python "$ROOT/export_human_study.py" \
    --system-a "$BASE_SELECT_FILE" \
    --system-b "$SYSTEM_B" \
    --raw-data "$RAW_DATA" \
    --output "$HUMAN_EXPORT"
else
  echo "Skipping human export; provide SYSTEM_B to compare against another system output."
fi

# Optional robustness on iterative loop baseline if available
LOOP_OUTPUT="${LOOP_OUTPUT:-$ROOT/results/loop.jsonl}"
if [[ -s "$LOOP_OUTPUT" ]]; then
  run_robustness "$LOOP_OUTPUT" "iterative_loop_baseline" "$TAU"
fi

echo "All FinalProposal runs finished. Outputs under $OUT_DIR"
