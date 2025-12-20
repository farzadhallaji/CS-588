# Experiment Protocol (for the paper)

This document states what each experiment is, why it exists, and where its results live. The goal is reproducible, paper-ready reporting, not implementation details.

## Data and Metrics
- **Dataset**: CRScore human study phase 1 (`raw_data.json`), providing diffs, seed reviews, and pseudo-reference claims.
- **Metrics**: CRScore Conciseness (Con), Comprehensiveness (Comp), and their F1 Relevance (Rel). Default similarity threshold τ = 0.7314 (per CRScore paper). Evidence guardrail τ_evidence = 0.35. All summaries report Con/Comp/Rel by language and overall.

## Systems Evaluated
1) **Human seed baseline**: unchanged human review.
2) **Iterative loop (ours)**: multi-iteration editing guided by CRScore selection and evidence guardrails.
3) **Single-pass edit (k1)**: one-shot edit with the same inputs, to isolate iteration.
4) **Single-pass rewrite**: one-shot full rewrite (no minimal-edit constraint).
5) **Random selection ablation**: loop without CRScore-based selection.
6) **No-evidence ablation**: loop without the evidence guardrail.
7) **Iterative rewrite**: loop with full rewrites each iteration.
8) **Proposal v1 few-shot baseline**: prompt-based baseline using mined few-shot pairs; run across multiple local models.
9) **Threshold-gated refinement**: if seed Rel < 0.6, run one local LLM refinement; evaluated across multiple models and prompt variants.
10) **Optional HF local threshold run**: same as (9) with a local HF model (Qwen) when available.

## Editors and Prompt Variants
- **Local models (Ollama)**: default `llama3:8b-instruct-q4_0`; additional defaults `deepseek-coder:6.7b-base-q4_0`, `qwen2.5-coder:7b` for multi-model sweeps.
- **Threshold prompt variants** (each crossed with every model):  
  - `default`: general improvement.  
  - `concise`: 1–3 sentences, must mention main change + one concrete test.  
  - `evidence`: add only claims-supported points; otherwise suggest a test.  
  - `test-heavy`: emphasize test coverage and edge cases.

## Experimental Conditions and Outputs
All results are written to `results/` (or `results_hf/` for the HF threshold run). JSONL files contain per-instance scores; summaries provide per-language and overall averages.

- **Human seed baseline**: `baseline_summary.json`.
- **Iterative loop (ours)**: `loop.jsonl` + `loop_summary.json`.
- **Single-pass edit (k1)**: `single_edit.jsonl` + `single_edit_summary.json`.
- **Single-pass rewrite**: `single_rewrite.jsonl` + `single_rewrite_summary.json`.
- **Random selection ablation**: `no_selection.jsonl` + `no_selection_summary.json`.
- **No-evidence ablation**: `no_evidence.jsonl` + `no_evidence_summary.json`.
- **Iterative rewrite**: `rewrite_loop.jsonl` + `rewrite_loop_summary.json`.
- **Proposal v1 few-shot baseline (per model)**: `proposal_v1_<model>.jsonl` + `proposal_v1_<model>_summary.json`.
- **Threshold-gated refinement (per model × prompt)**: `threshold_<prompt>_<model>.jsonl` + `threshold_<prompt>_<model>_summary.json`. Includes both seed and final Con/Comp/Rel plus an `improved` flag.
- **HF threshold run (optional)**: `threshold_qwen_evidence.jsonl` + `_summary.json` in `results_hf/`.
- **Human evaluation stats (phase 2)**: `human_eval_summary.json` (paired statistics against human ratings when the phase 2 files and SciPy are available).

## Intended Tables/Figures (paper mapping)
- **Main results table**: human seed vs iterative loop vs single-pass edit vs single-pass rewrite, reported on test split with Con/Comp/Rel; include human phase 2 stats if available.
- **Ablation table**: iterative loop vs random selection vs no-evidence vs iterative rewrite vs k1; same metrics.
- **Threshold sweep table**: per prompt/model threshold refinement outcomes (seed vs refined Rel/Con/Comp, improvement rate).
- **Model sweep (proposal v1)**: per-model performance for the few-shot baseline.
- **Optional figure**: quality vs cost frontier from varying loop parameters (iterations/samples) or comparing small local models.

## How to Interpret Results
- **Con/Comp/Rel** appear in all summaries; Rel is the primary headline, with Con/Comp showing precision/recall trade-offs.
- **Threshold runs** explicitly show both seed and post-refinement scores to quantify gains and improvement rate.
- **Ablations** demonstrate the causal contribution of iteration, selection, evidence, and rewrite vs edit framing.

## Reproducibility Notes
- The pipeline is idempotent: existing outputs are reused; delete a file to rerun that experiment.
- All CRScore evaluations use τ = 0.7314. Refinement gating uses Rel < 0.6 by design.
- Multi-model sweeps always save distinct files per model and prompt, enabling direct comparison without re-running. 
