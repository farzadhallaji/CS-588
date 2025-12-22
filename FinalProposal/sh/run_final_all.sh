#!/usr/bin/env bash
set -euo pipefail

# FinalProposal end-to-end runner with ablations + resumability.
# Produces:
# - Reward rerank baseline (candidates -> select -> evaluate -> robustness)
# - Generation ablations (prompt variants, N samples, temperature)
# - Reward ablations (penalties on/off, tau/temp sweeps, hard CRScore)
# - Preference dumps (top1/bottom1 vs top2/bottom2, with/without evidence penalty)
# - Optional DPO LoRA train/infer (set DPO_MODELS to enable)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATA="${RAW_DATA:-$ROOT/../CRScore/human_study/phase1/raw_data.json}"
TAU="${TAU:-0.7314}"
MODEL_PATH="${MODEL_PATH:-mixedbread-ai/mxbai-embed-large-v1}"
MODEL_TYPE="${MODEL_TYPE:-ollama}" # ollama|hf-local|echo|lora
MODEL_NAME="${MODEL_NAME:-llama3:8b-instruct-q4_0}"
PROMPTS="${PROMPTS:-default,evidence_grounded,test_heavy,concise}"
OUT_DIR="${OUT_DIR:-$ROOT/results}"
CAND_DIR="${CAND_DIR:-$OUT_DIR/candidates}"
SEL_DIR="${SEL_DIR:-$OUT_DIR/selected}"
SUM_DIR="${SUM_DIR:-$OUT_DIR/summaries}"
PREF_DIR="${PREF_DIR:-$OUT_DIR/preferences}"
ROBUST_DIR="${ROBUST_DIR:-$OUT_DIR/robustness}"
LORA_DIR="${LORA_DIR:-$OUT_DIR/lora}"
SYSTEM_B="${SYSTEM_B:-}" # optional: export human overlap vs another system output

mkdir -p "$OUT_DIR" "$CAND_DIR" "$SEL_DIR" "$SUM_DIR" "$PREF_DIR" "$ROBUST_DIR" "$LORA_DIR"

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
  else
    echo "ollama serve already running."
  fi
  if ! ollama list >/dev/null 2>&1; then
    echo "ollama serve did not become ready; check /tmp/ollama-serve.log" >&2
    exit 1
  fi
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

gen_candidates() {
  local cfg="$1"
  local name="" num_samples=2 temperature=0.3 prompts="$PROMPTS" limit=""
  for kv in $cfg; do
    local k="${kv%%=*}" v="${kv#*=}"
    eval "$k=\"$v\""
  done
  local cand_file="$CAND_DIR/candidates_${name}.jsonl"
  run_or_skip "$cand_file" python "$ROOT/generate_candidates.py" \
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
  run_or_skip "$select_file" python "$ROOT/select_best.py" \
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

  run_or_skip "$summary_file" python "$ROOT/../evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$tau_val" \
    --model-path "$MODEL_PATH" \
    --outputs "$select_file" \
    --summary-out "$summary_file"

  echo "$select_file"
}

run_robustness() {
  local select_file="$1"
  local tag="$2"
  local tau_val="${3:-$TAU}"
  local rob_file="$ROBUST_DIR/robust_${tag}.csv"
  run_or_skip "$rob_file" python "$ROOT/robustness_suite.py" \
    --outputs "$select_file" \
    --raw-data "$RAW_DATA" \
    --tau "$tau_val" \
    --model-path "$MODEL_PATH" \
    --output-csv "$rob_file"
}

build_pref_variant() {
  local cand_file="$1"
  local cand_tag="$2"
  local cfg="$3"
  local name="" tau_val="$TAU" temp=0.05 evidence_margin=0.35 top_k_align=2 score_mode="soft"
  local w_rel=1.0 w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400
  local top_k=1 bottom_k=1 include_median=0
  for kv in $cfg; do
    local k="${kv%%=*}" v="${kv#*=}"
    eval "$k=\"$v\""
  done
  local pref_file="$PREF_DIR/prefs_${cand_tag}__${name}.jsonl"
  if [[ -s "$pref_file" ]]; then
    echo "Skipping; found $pref_file"
    echo "$pref_file"
    return
  fi
  args=(
    "$ROOT/build_preferences.py"
    --candidates "$cand_file"
    --tau "$tau_val"
    --temp "$temp"
    --evidence-margin "$evidence_margin"
    --model-path "$MODEL_PATH"
    --score-mode "$score_mode"
    --top-k-align "$top_k_align"
    --top-k "$top_k"
    --bottom-k "$bottom_k"
    --w-rel "$w_rel"
    --w-unsupported "$w_unsupported"
    --w-len "$w_len"
    --w-copy "$w_copy"
    --len-norm "$len_norm"
    --output "$pref_file"
  )
  if [[ "$include_median" == "1" ]]; then
    args+=(--include-median)
  fi
  run_or_skip "$pref_file" python "${args[@]}"
  echo "$pref_file"
}

check_dpo_deps() {
  python - <<'PY'
try:
    import trl  # noqa: F401
    import peft  # noqa: F401
    import datasets  # noqa: F401
    import transformers  # noqa: F401
    ok = True
except Exception:
    ok = False
exit(0 if ok else 1)
PY
}

train_dpo_lora() {
  local prefs="$1"
  local model="$2"
  local tag="$3"
  local out_dir="$LORA_DIR/${tag}__$(slug "$model")"
  local sentinel="$out_dir/adapter_config.json"
  if [[ -s "$sentinel" ]]; then
    echo "Skipping DPO train; found $sentinel"
    echo "$out_dir"
    return
  fi
  mkdir -p "$out_dir"
  run_or_skip "$sentinel" python "$ROOT/train_dpo_lora.py" \
    --prefs "$prefs" \
    --model-name "$model" \
    --output-dir "$out_dir" \
    --max-steps "${DPO_MAX_STEPS:-200}" \
    --per-device-batch-size "${DPO_BATCH:-1}" \
    --grad-accum "${DPO_GRAD_ACCUM:-4}" \
    --learning-rate "${DPO_LR:-5e-5}" \
    --lora-r "${DPO_LORA_R:-16}" \
    --lora-alpha "${DPO_LORA_ALPHA:-32}" \
    --lora-dropout "${DPO_LORA_DROPOUT:-0.05}" \
    --beta "${DPO_BETA:-0.1}"
  echo "$out_dir"
}

infer_lora() {
  local lora_path="$1"
  local base_model="$2"
  local tag="$3"
  local select_file="$SEL_DIR/lora_${tag}__$(slug "$base_model").jsonl"
  local summary_file="$SUM_DIR/lora_summary_${tag}__$(slug "$base_model").json"
  run_or_skip "$select_file" python "$ROOT/infer_final.py" \
    --mode generate \
    --raw-data "$RAW_DATA" \
    --split test \
    --num-samples "${LORA_NUM_SAMPLES:-2}" \
    --prompt-variants "$PROMPTS" \
    --model-type lora \
    --model-name "$base_model" \
    --lora-base "$base_model" \
    --lora-path "$lora_path" \
    --temperature "${LORA_TEMPERATURE:-0.3}" \
    --model-path "$MODEL_PATH" \
    --tau "$TAU" \
    --temp "${LORA_SOFT_TEMP:-0.05}" \
    --score-mode "${LORA_SCORE_MODE:-soft}" \
    --output "$select_file"

  run_or_skip "$summary_file" python "$ROOT/../evaluate.py" \
    --raw-data "$RAW_DATA" \
    --split test \
    --tau "$TAU" \
    --model-path "$MODEL_PATH" \
    --outputs "$select_file" \
    --summary-out "$summary_file"

  run_robustness "$select_file" "lora_${tag}__$(slug "$base_model")" "$TAU"
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

PREF_CONFIGS=(
  "name=top1_ev score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU} top_k=1 bottom_k=1 include_median=1"
  "name=top1_noev score_mode=soft w_unsupported=0 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU} top_k=1 bottom_k=1"
  "name=top2_ev score_mode=soft w_unsupported=0.6 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU} top_k=2 bottom_k=2"
  "name=top2_noev score_mode=soft w_unsupported=0 w_len=0.02 w_copy=0.15 len_norm=400 temp=0.05 tau=${TAU} top_k=2 bottom_k=2"
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

echo "=== Build DPO preference pairs ==="
for cfg in "${PREF_CONFIGS[@]}"; do
  pref_name=""
  for kv in $cfg; do
    k="${kv%%=*}"; v="${kv#*=}"; [[ "$k" == "name" ]] && pref_name="$v"
  done
  [[ -z "$pref_name" ]] && pref_name="prefs"
  build_pref_variant "$BASE_CAND_FILE" "base" "$cfg" >/dev/null
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

echo "=== Optional: DPO LoRA (set DPO_MODELS to enable) ==="
if [[ -n "${DPO_MODELS:-}" ]]; then
  if check_dpo_deps; then
    IFS=',' read -r -a DPO_MODEL_LIST <<<"$DPO_MODELS"
    DPO_PREF_TAG="${DPO_PREF_TAG:-top1_ev}"
    DPO_PREF_FILE="$PREF_DIR/prefs_base__${DPO_PREF_TAG}.jsonl"
    if [[ ! -s "$DPO_PREF_FILE" ]]; then
      echo "Requested DPO_PREF_TAG=${DPO_PREF_TAG} but file missing: $DPO_PREF_FILE" >&2
    else
      for MODEL in "${DPO_MODEL_LIST[@]}"; do
        MODEL_TRIMMED="$(echo "$MODEL" | xargs)"
        [[ -z "$MODEL_TRIMMED" ]] && continue
        LORA_PATH=$(train_dpo_lora "$DPO_PREF_FILE" "$MODEL_TRIMMED" "$DPO_PREF_TAG")
        infer_lora "$LORA_PATH" "$MODEL_TRIMMED" "$DPO_PREF_TAG"
      done
    fi
  else
    echo "Skipping DPO; install trl/peft/datasets/transformers first."
  fi
else
  echo "DPO_MODELS not set; skipping LoRA training/inference."
fi

# Optional robustness on iterative loop baseline if available
LOOP_OUTPUT="${LOOP_OUTPUT:-$ROOT/results/loop.jsonl}"
if [[ -s "$LOOP_OUTPUT" ]]; then
  run_robustness "$LOOP_OUTPUT" "iterative_loop_baseline" "$TAU"
fi

echo "All FinalProposal runs finished. Outputs under $OUT_DIR"
