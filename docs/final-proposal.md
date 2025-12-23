# Review reward-rerank pipeline

Two-stage system to turn a “junior” code review into a stronger one: first generate multiple rewrites, then score and pick the best, followed by evaluation and robustness checks.

## Overview
- Data: fixed set of review items (diff + junior review + “claims” that describe what a good review should cover). Default run uses the test split of 120 items (40 per language).
- Two stages:
  1) **Generate** several candidate rewrites with different prompt styles and sampling settings.
  2) **Select** the best candidate using a reward that favors relevance to the claims and penalizes bad behaviors (unsupported statements, verbosity, copying).
- After selection: report metrics and run robustness perturbations to see how brittle the chosen outputs are.

## Candidate generation
- What happens: for each item, the model rewrites the review multiple times. Prompt style and sampling temperature are varied to produce a spread of candidates.
- Models: default is an Ollama-served model similar to llama3 8B; you can swap to other local models or adapters.
- Prompt styles:
  - **default**: general improvement aligned to the change.
  - **evidence_grounded**: only include points the claims support; suggest tests otherwise.
  - **test_heavy**: emphasize concrete tests and failure modes.
  - **concise**: short review with main change plus a check/test.
- Sampling knobs: number of samples per item and randomness (temperature). Defaults are 2 samples at temperature 0.3; ablations try more samples (4, 8) and temperatures 0.2 and 0.6.

Exact prompt texts (what the model sees before the review/claims block):
- **default**  
  Improve the review so it better matches the claims.  
  Include missing important points, remove irrelevant content.  
  If something is uncertain, phrase it as a verification/test request.  
  Output only the revised review.

- **evidence_grounded**  
  Add ONLY points directly supported by the claims.  
  If support is missing, do not add the point; suggest a test instead.  
  Keep it short and specific.  
  Output only the revised review.

- **test_heavy**  
  Focus on test coverage and failure modes implied by the claims.  
  Add 1-2 concrete tests (happy path + edge case).  
  Avoid restating obvious change details unless needed.  
  Output only the revised review.

- **concise**  
  Rewrite the review into 1-3 sentences.  
  Must mention the main change and one concrete check/test.  
  Remove everything else.  
  Output only the revised review.

## Reward selection
- What happens: every candidate is scored against the claims using an embedding-based relevance measure. The score starts from relevance and subtracts penalties.
- Scoring formula (conceptual):  
  `reward = w_rel * relevance – w_unsupported * unsupported – w_len * length_penalty – w_copy * copy_penalty`  
  - **relevance**: how well the candidate matches the claims.  
  - **unsupported**: how much it asserts things not backed by claims.  
  - **length_penalty**: grows when the review is longer than a target length.  
  - **copy_penalty**: punishes parroting input text.  
- Soft vs hard: soft uses the continuous relevance score; hard treats matches as pass/fail at a similarity threshold. Default threshold is 0.7314.
- Default weights: relevance 1.0, unsupported 0.6, length 0.02, copy 0.15, target length 400 tokens, selection temperature 0.05, alignment depth 2, evidence margin 0.35.
- Why penalties: to curb hallucinations, verbosity, and copying while keeping alignment to the claims.
- Output: one best review per item, plus its scores.

## Robustness
- After selecting the best reviews, we perturb inputs (e.g., tweak text) and see how scores change. The goal is to check stability, not just peak scores.

## Outputs and summaries
- Candidates: all generated rewrites, tagged by prompt and sampling settings.
- Selected: the chosen best review per item, plus scores.
- Robustness: tables of how scores shift under perturbations.

## Key knobs
- Generation: prompt style, number of samples, temperature, model choice.
- Selection: score mode (soft or hard threshold), similarity threshold, penalty weights (unsupported/length/copy), selection temperature, target length, alignment depth, evidence margin.
- Data: which split and how many items to process; which embedding model to use.

## Experiments/ablations that were run (per run_final_all.sh)
- Base: `num_samples=2`, `temperature=0.3`, prompts=`default,evidence_grounded,test_heavy,concise`, model `llama3:8b-instruct-q4_0`, reward config `reward_default` (soft + penalties).
- Generation ablations:
  - Temps: `temp02` (0.2), `temp06` (0.6).
  - Sample count: `num4` (4), `num8` (8).
  - Prompt-only: `prompt_default`, `prompt_evidence`, `prompt_concise`, `prompt_testheavy`.
- Score/reward ablations (all on base candidates):
  - Penalty toggles: `soft_only`, `soft_len`, `soft_evidence`, `soft_evidence_len`, `soft_evidence_len_copy` (full).
  - Mode: `hard_cr`.
  - Tau sweeps: `tau_low=0.65`, `tau_high=0.80`.
  - Selection temp sweeps: `soft_temp_low=0.02`, `soft_temp_high=0.10`.
- Robustness: run for base selection; optional for iterative loop baseline when `results/loop.jsonl` exists.

## Why this setup (plain English, no code needed)
- Two-stage design: we first **generate** multiple candidate reviews with different prompts/temperatures to cover diverse rewrites, then **select** using a scoring function that rewards alignment to references (claims) and penalizes bad behaviors (unsupported statements, overly long text, copying). This separates creativity (LLM sampling) from discipline (scoring).
- Scoring math (conceptually): each candidate gets
  - a relevance/quality score from CRScore (embedding similarity vs pseudo-refs), gated by `tau` (the match threshold),
  - minus weighted penalties: unsupported claims (evidence gap), excessive length (normalize by `len_norm`), and copy ratio (if it just parrots the input),
  - combined as `reward = w_rel * rel_score – w_unsupported * unsupported – w_len * length_penalty – w_copy * copy_penalty`.
  Soft mode uses the continuous score; hard mode uses a binary pass/fail at tau.
- Why tau sweeps: moving `tau` changes how strict we are about what counts as “matched” to a claim. Lower tau is forgiving (more candidates pass), higher tau is strict (fewer, higher-precision matches).
- Why selection temperature: even after scoring, small randomness (`temp`) in picking the best can avoid ties or overfitting to one noisy score; sweeping it shows sensitivity.
- Why prompt/temperature/num-sample sweeps: more samples or higher temp increase exploration (may find better rewrites but more noise). Prompt variants bias the generator toward evidence grounding, brevity, or test focus; we compare these directions.
- Why penalties: unsupported and copy penalties reduce hallucinations and trivial copying; length penalty keeps reviews tight. Toggling them shows which constraints actually help metrics.
- Robustness: after picking the “best,” we perturb inputs to see how much scores degrade—testing stability, not just raw performance.
