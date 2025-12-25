# Reproduction Guide

Data requirements, configuration entry points, and step-by-step commands to regenerate the main results.

## Data
- Required: `CRScore-human_study/phase1/raw_data.json` (now in-repo). Set `RAW_DATA=$PWD/CRScore-human_study/phase1/raw_data.json` or point to another copy.
- Optional: `enhanced_code_reviews_with_index.csv` (in repo) is only needed for per-review CRScore analysis (`scripts/compute_per_review_crscore.py`).
- Extra (not needed for the main runs): pilot and phase2 annotation CSVs plus context files under `CRScore-human_study/pilot_v2/`, `CRScore-human_study/pilot2/`, and `CRScore-human_study/phase2/`.

## Environment setup
- Python 3.10+ recommended. Example:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -U sentence-transformers torch transformers requests pandas tqdm numpy
  ```
- Local LLMs: the main runs expect Ollama models. Pull the defaults or swap via env vars:
  ```bash
  ollama pull llama3:8b-instruct-q4_0
  ollama pull deepseek-coder:6.7b-base-q4_0
  ollama pull qwen2.5-coder:7b
  ```
- HF model link for the Magicoder variant used in ablations: https://huggingface.co/teledia-cmu/Magicoder-6.7B-code-change-summ-impl (download if you want HF-local generation instead of Ollama).
- Embeddings: defaults use `mixedbread-ai/mxbai-embed-large-v1` (HF link: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1). It is pulled automatically on first run; set `MODEL_PATH` to a local copy if needed.

## Models used (with sources)
- `mixedbread-ai/mxbai-embed-large-v1` — embedding model for CRScore scoring (HF).
- `llama3:8b-instruct-q4_0` — default Ollama model for generation (pull via `ollama pull llama3:8b-instruct-q4_0`; base model on HF: https://huggingface.co/meta-llama/Meta-Llama-3-8B).
- `deepseek-coder:6.7b-base-q4_0` — Ollama quantized DeepSeek Coder (upstream HF: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base).
- `qwen2.5-coder:7b` — Ollama quantized Qwen2.5 Coder 7B (upstream HF: https://huggingface.co/Qwen/Qwen2.5-Coder-7B).
- `teledia-cmu/Magicoder-6.7B-code-change-summ-impl` — HF model used in HF-local ablations / offline runs.
- `EchoEditor` (no model) — used only for ablations.

## Key configuration files
- `run_all.sh`: single entry point that runs baselines, iterative loop ablations, threshold-gated refinement sweeps, and the reward-rerank pipeline. Tunables are exposed as env vars (`RAW_DATA`, `SPLIT`, `TAU`, `MODEL_PATH`, `OLLAMA_MODEL`, `EXTRA_OLLAMA_MODELS`, `THRESHOLD_PROMPTS`, etc.).
- `review_reward_rerank/run_reward_rerank.sh`: focused runner for the two-stage reward-rerank system with generation and scoring ablations.
- Component scripts (invoked by the runners): `scripts/make_splits.py`, `scripts/build_fewshot_pairs.py`, `run.py`, `threshold_refine.py`, `proposal_v1.py`, `evaluate.py`, and `review_reward_rerank/*.py`.

## Reproducing the main results
1) **Point to the data**  
   ```bash
   export RAW_DATA=$PWD/CRScore-human_study/phase1/raw_data.json
   ```
   Optionally set `SPLIT=test` (default), `OUT_DIR=results`, and `RERANK_OUT=results/reward_rerank`.

2) **Ensure models are available**  
   Start the Ollama daemon if needed (`ollama serve`) and pull the models listed above, or set `OLLAMA_MODEL`/`EXTRA_OLLAMA_MODELS` to ones you have locally. If you prefer HF-local generation, set `RERANK_MODEL_TYPE=hf-local` and `RERANK_MODEL_NAME=/path/to/model`.

3) **Run the full suite**  
   ```bash
   bash run_all.sh
   ```
   What it does (skipping steps whose outputs already exist):
   - Freezes dev/test splits (`scripts/make_splits.py`) and builds few-shot pairs.
   - Scores the human baseline and runs the iterative refinement loop variants (`run.py` + `evaluate.py`).
   - Sweeps threshold-gated single-pass rewrites over prompt/model variants (`threshold_refine.py` + `evaluate.py`).
   - Runs the reward-rerank pipeline: candidate generation, reward selection, evaluation, robustness checks.

4) **Inspect outputs**  
   - Baselines, loops, and threshold runs: `results/<split>/baseline/*.json[l]` and summary JSONs in `results/..._summary.json`.
   - Reward-rerank: candidates/selected/summaries/robustness under `results/<split>/reward_rerank/`.
   - Key papers’ numbers come from the threshold summaries (e.g., `results/threshold_concise_deepseek-coder_6.7b-base-q4_0_summary.json`) and reward-rerank summaries (e.g., `results/<split>/reward_rerank/summaries/summary_base__reward_default.json`).

## Minimal rerun for reward-rerank only
If you only need the two-stage system, run:
```bash
export RAW_DATA=/path/to/raw_data.json
bash review_reward_rerank/run_reward_rerank.sh
```
Use the env vars inside that script to adjust models, prompt variants, and scoring weights.
