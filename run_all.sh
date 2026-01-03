#!/usr/bin/env bash
set -euo pipefail

# Unified runner: baselines, ablations, proposal v1, threshold refinement, and reward-rerank pipeline.
# Configure via env vars as needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Shared defaults
RAW_DATA="${RAW_DATA:-$ROOT/../CRScore/human_study/phase1/raw_data.json}"
SPLIT="${SPLIT:-all}" # dev|test|all
MODEL_PATH="${MODEL_PATH:-mixedbread-ai/mxbai-embed-large-v1}"
TAU="${TAU:-0.7314}"
TAU_EVIDENCE="${TAU_EVIDENCE:-0.35}"
MAX_CHANGE="${MAX_CHANGE:-0.7}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3:8b-instruct-q4_0}"
EXTRA_OLLAMA_MODELS="${EXTRA_OLLAMA_MODELS:-deepseek-coder:6.7b-base-q4_0,qwen2.5-coder:7b}" # comma-separated
THRESHOLD_PROMPTS="${THRESHOLD_PROMPTS:-default,concise,evidence,test-heavy}"
THETA_SENS_OUT="${THETA_SENS_OUT:-$OUT_DIR/$SPLIT/theta_sensitivity}"
EVIDENCE_SENS_OUT="${EVIDENCE_SENS_OUT:-$OUT_DIR/$SPLIT/evidence_sensitivity}"
ITER_SENS_OUT="${ITER_SENS_OUT:-$OUT_DIR/$SPLIT/iter_sensitivity}"

# Output roots
OUT_DIR="${OUT_DIR:-$ROOT/results}"
BASE_OUT="${BASE_OUT:-$OUT_DIR/$SPLIT/baseline}"
RERANK_OUT="${RERANK_OUT:-$OUT_DIR/$SPLIT/reward_rerank}"
HF_OUT_DIR="${HF_OUT_DIR:-$ROOT/results_hf}"

# Baseline/proposal outputs
SPLITS_OUT="${SPLITS_OUT:-$BASE_OUT/splits.json}"
FEWSHOT_OUT="${FEWSHOT_OUT:-$BASE_OUT/fewshot_pairs.json}"
PROPOSAL_OUT="${PROPOSAL_OUT:-$BASE_OUT/proposal_v1.jsonl}"
PROPOSAL_SUMMARY="${PROPOSAL_SUMMARY:-$BASE_OUT/proposal_v1_summary.json}"

# Reward-rerank defaults
RERANK_MODEL_TYPE="${RERANK_MODEL_TYPE:-ollama}" # ollama|hf-local|echo|lora
RERANK_MODEL_NAME="${RERANK_MODEL_NAME:-llama3:8b-instruct-q4_0}"
RERANK_PROMPTS="${RERANK_PROMPTS:-default,evidence_grounded,test_heavy,concise}"
RERANK_CAND_DIR="${RERANK_CAND_DIR:-$RERANK_OUT/candidates}"
RERANK_SEL_DIR="${RERANK_SEL_DIR:-$RERANK_OUT/selected}"
RERANK_SUM_DIR="${RERANK_SUM_DIR:-$RERANK_OUT/summaries}"
RERANK_ROBUST_DIR="${RERANK_ROBUST_DIR:-$RERANK_OUT/robustness}"
SYSTEM_B="${SYSTEM_B:-}" # optional export for human study comparison

mkdir -p "$OUT_DIR" "$BASE_OUT" "$RERANK_OUT" "$RERANK_CAND_DIR" "$RERANK_SEL_DIR" "$RERANK_SUM_DIR" "$RERANK_ROBUST_DIR"

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

slug() {
  echo "$1" | tr '/:' '__' | tr ' ' '_' | tr '[:upper:]' '[:lower:]'
}

# ---------------- Baselines + proposal v1 + threshold sweep ----------------
echo "=== Freeze deterministic dev/test split ==="
run_or_skip "$SPLITS_OUT" python "$ROOT/scripts/make_splits.py" --raw-data "$RAW_DATA" --out "$SPLITS_OUT"


echo "=== Build few-shot pairs for proposal v1 baseline ==="
run_or_skip "$FEWSHOT_OUT" python "$ROOT/scripts/build_fewshot_pairs.py" --raw-data "$RAW_DATA" --tau "$TAU" --model-path "$MODEL_PATH" --out "$FEWSHOT_OUT"

echo "=== Baseline (human seed) ==="
run_or_skip "$BASE_OUT/baseline_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --baseline-only --summary-out "$BASE_OUT/baseline_summary.json"

echo "=== Main loop + ablations (template editor) ==="
run_or_skip "$BASE_OUT/loop.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split "$SPLIT" --mode loop --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$BASE_OUT/loop.jsonl"
run_or_skip "$BASE_OUT/single_edit.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split "$SPLIT" --mode k1 --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$BASE_OUT/single_edit.jsonl"
run_or_skip "$BASE_OUT/single_rewrite.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split "$SPLIT" --mode rewrite --max-iter 1 --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change 1.0 --output "$BASE_OUT/single_rewrite.jsonl"
run_or_skip "$BASE_OUT/no_selection.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split "$SPLIT" --mode no-selection --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$BASE_OUT/no_selection.jsonl"
run_or_skip "$BASE_OUT/no_evidence.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split "$SPLIT" --mode no-evidence --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change "$MAX_CHANGE" --output "$BASE_OUT/no_evidence.jsonl"
run_or_skip "$BASE_OUT/rewrite_loop.jsonl" python "$ROOT/run.py" --raw-data "$RAW_DATA" --split "$SPLIT" --mode rewrite --model-path "$MODEL_PATH" --tau "$TAU" --tau-evidence "$TAU_EVIDENCE" --max-change 1.0 --output "$BASE_OUT/rewrite_loop.jsonl"

echo "=== Evaluate loop outputs ==="
run_or_skip "$BASE_OUT/loop_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$BASE_OUT/loop.jsonl" --summary-out "$BASE_OUT/loop_summary.json"
run_or_skip "$BASE_OUT/single_edit_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$BASE_OUT/single_edit.jsonl" --summary-out "$BASE_OUT/single_edit_summary.json"
run_or_skip "$BASE_OUT/single_rewrite_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$BASE_OUT/single_rewrite.jsonl" --summary-out "$BASE_OUT/single_rewrite_summary.json"
run_or_skip "$BASE_OUT/no_selection_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$BASE_OUT/no_selection.jsonl" --summary-out "$BASE_OUT/no_selection_summary.json"
run_or_skip "$BASE_OUT/no_evidence_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$BASE_OUT/no_evidence.jsonl" --summary-out "$BASE_OUT/no_evidence_summary.json"
run_or_skip "$BASE_OUT/rewrite_loop_summary.json" python "$ROOT/evaluate.py" --raw-data "$RAW_DATA" --split "$SPLIT" --tau "$TAU" --model-path "$MODEL_PATH" --outputs "$BASE_OUT/rewrite_loop.jsonl" --summary-out "$BASE_OUT/rewrite_loop_summary.json"

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
    OUT_FILE="$BASE_OUT/proposal_v1_${MODEL_SLUG}.jsonl"
    SUM_FILE="$BASE_OUT/proposal_v1_${MODEL_SLUG}_summary.json"
    run_or_skip "$OUT_FILE" python "$ROOT/proposal_v1.py" \
      --raw-data "$RAW_DATA" \
      --split "$SPLIT" \
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
      OUT_FILE="$BASE_OUT/threshold_${PV_TRIMMED}_${MODEL_SLUG}.jsonl"
      SUM_FILE="$BASE_OUT/threshold_${PV_TRIMMED}_${MODEL_SLUG}_summary.json"
      run_or_skip "$OUT_FILE" python "$ROOT/threshold_refine.py" \
        --raw-data "$RAW_DATA" \
        --split "$SPLIT" \
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

echo "=== Ablations: θ gate sensitivity (0.5/0.6/0.7) ==="
if ensure_ollama; then
  if ensure_model "$OLLAMA_MODEL"; then
    run_or_skip "$THETA_SENS_OUT/refine_summary.json" python "$ROOT/scripts/threshold_sensitivity.py" \
      --raw-data "$RAW_DATA" \
      --split "$SPLIT" \
      --thresholds 0.5 0.6 0.7 \
      --prompt-variant default \
      --model-type ollama \
      --model-name "$OLLAMA_MODEL" \
      --run-refine \
      --output-dir "$THETA_SENS_OUT"
  fi
else
  echo "Skipping θ sensitivity; ollama not available."
fi

echo "=== Ablations: τ_evidence sensitivity (0.25/0.35/0.45) ==="
if ensure_ollama; then
  if ensure_model "$OLLAMA_MODEL"; then
    run_or_skip "$EVIDENCE_SENS_OUT/summary.json" python "$ROOT/scripts/evidence_sensitivity.py" \
      --raw-data "$RAW_DATA" \
      --split "$SPLIT" \
      --tau-evidence 0.25 0.35 0.45 \
      --max-iter 3 \
      --num-samples 2 \
      --prompt-style loop \
      --selection crscore \
      --model-type ollama \
      --model-name "$OLLAMA_MODEL" \
      --output-dir "$EVIDENCE_SENS_OUT"
  fi
else
  echo "Skipping τ_evidence sensitivity; ollama not available."
fi

echo "=== Ablations: iteration budget sensitivity (N=1/2/3, K=2) ==="
if ensure_ollama; then
  if ensure_model "$OLLAMA_MODEL"; then
    run_or_skip "$ITER_SENS_OUT/summary.json" python "$ROOT/scripts/iter_sensitivity.py" \
      --raw-data "$RAW_DATA" \
      --split "$SPLIT" \
      --iters 1 2 3 \
      --num-samples 2 \
      --prompt-style loop \
      --selection crscore \
      --model-type ollama \
      --model-name "$OLLAMA_MODEL" \
      --output-dir "$ITER_SENS_OUT"
  fi
else
  echo "Skipping iteration sensitivity; ollama not available."
fi

echo "=== Stats: paired tests and CIs ==="
STATS_OUT="${STATS_OUT:-$OUT_DIR/$SPLIT/stats}"
mkdir -p "$STATS_OUT"
DEFAULT_THRESH_FILE="$BASE_OUT/threshold_default_$(slug "$OLLAMA_MODEL").jsonl"
if [[ -f "$DEFAULT_THRESH_FILE" ]]; then
  run_or_skip "$STATS_OUT/threshold_default_within.json" python "$ROOT/scripts/stats_tests.py" \
    --within-file "$DEFAULT_THRESH_FILE" \
    --output-dir "$STATS_OUT"
fi
# Optional model comparison: base model vs first extra on default prompt
if [[ -n "$EXTRA_OLLAMA_MODELS" ]]; then
  IFS=',' read -r -a EXTRA_MODELS <<<"$EXTRA_OLLAMA_MODELS"
  if [[ ${#EXTRA_MODELS[@]} -gt 0 ]]; then
    COMP_MODEL="$(echo "${EXTRA_MODELS[0]}" | xargs)"
    COMP_THRESH_FILE="$BASE_OUT/threshold_default_$(slug "$COMP_MODEL").jsonl"
    if [[ -n "$COMP_MODEL" && -f "$COMP_THRESH_FILE" && -f "$DEFAULT_THRESH_FILE" ]]; then
      run_or_skip "$STATS_OUT/threshold_default_$(slug "$OLLAMA_MODEL")_vs_$(slug "$COMP_MODEL").json" python "$ROOT/scripts/stats_tests.py" \
        --file-a "$DEFAULT_THRESH_FILE" \
        --file-b "$COMP_THRESH_FILE" \
        --label-a "$(slug "$OLLAMA_MODEL")" \
        --label-b "$(slug "$COMP_MODEL")" \
        --output-dir "$STATS_OUT"
    fi
  fi
fi

# ---------------- Reward rerank pipeline (review_reward_rerank) ----------------
gen_candidates() {
  local cfg="$1"
  local name="" num_samples=2 temperature=0.3 prompts="$RERANK_PROMPTS" limit=""
  for kv in $cfg; do
    local k="${kv%%=*}" v="${kv#*=}"
    eval "$k=\"$v\""
  done
  local cand_file="$RERANK_CAND_DIR/candidates_${name}.jsonl"
  run_or_skip "$cand_file" python -m review_reward_rerank.generate_candidates \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --model-type "$RERANK_MODEL_TYPE" \
    --model-name "$RERANK_MODEL_NAME" \
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
  local select_file="$RERANK_SEL_DIR/selected_${cand_tag}__${name}.jsonl"
  local summary_file="$RERANK_SUM_DIR/summary_${cand_tag}__${name}.json"
  run_or_skip "$select_file" python -m review_reward_rerank.select_best \
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

  run_or_skip "$summary_file" python "$ROOT/evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split "$SPLIT" \
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
  local rob_file="$RERANK_ROBUST_DIR/robust_${tag}.csv"
  run_or_skip "$rob_file" python -m review_reward_rerank.robustness_suite \
    --outputs "$select_file" \
    --raw-data "$RAW_DATA" \
    --tau "$tau_val" \
    --model-path "$MODEL_PATH" \
    --output-csv "$rob_file"
}

BASE_GEN_CFG="name=base num_samples=2 temperature=0.3 prompts=${RERANK_PROMPTS}"
GEN_ABLATIONS=(
  "name=temp02 num_samples=2 temperature=0.2 prompts=${RERANK_PROMPTS}"
  "name=temp06 num_samples=2 temperature=0.6 prompts=${RERANK_PROMPTS}"
  "name=num4 num_samples=4 temperature=0.3 prompts=${RERANK_PROMPTS}"
  "name=num8 num_samples=8 temperature=0.3 prompts=${RERANK_PROMPTS}"
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

echo "=== Reward rerank pipeline ==="
if [[ "$RERANK_MODEL_TYPE" == "ollama" ]] && ! ensure_ollama; then
  echo "Skipping reward rerank pipeline; ollama not available."
else
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
    HUMAN_EXPORT="${HUMAN_EXPORT:-$RERANK_OUT/human_export.csv}"
    python "$ROOT/review_reward_rerank/export_human_study.py" \
      --system-a "$BASE_SELECT_FILE" \
      --system-b "$SYSTEM_B" \
      --raw-data "$RAW_DATA" \
      --output "$HUMAN_EXPORT"
  else
    echo "Skipping human export; provide SYSTEM_B to compare against another system output."
  fi

  LOOP_OUTPUT="${LOOP_OUTPUT:-$BASE_OUT/loop.jsonl}"
  if [[ -s "$LOOP_OUTPUT" ]]; then
    run_robustness "$LOOP_OUTPUT" "iterative_loop_baseline" "$TAU"
  fi
fi

echo "All runs complete. Baseline outputs under $BASE_OUT, reward rerank under $RERANK_OUT"
