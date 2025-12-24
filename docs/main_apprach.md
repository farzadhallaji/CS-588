# Main Approach: Threshold-Gated Single-Pass Rewrite

Reader assumption: you know **CRScore** (Rel/Con/Comp in `[0,1]`) but nothing else.

## TL;DR
Score the seed review with CRScore. If its relevance (Rel) is below a threshold τ, rewrite it once with a small local LLM using a structured prompt. Re-score; keep whichever version scores higher. This simple gate → rewrite → re-score loop delivers 4–5× quality gains over baseline at low cost.

## Problem Setup
- **Inputs**: code diff, seed (junior) review, and claims describing what a good review should cover.
- **Hard CRScore (what drives gating)**: Rel/Con/Comp in `[0,1]` from discrete matches to the claims. We trigger the rewrite when Rel < τ, and we keep the higher-Rel version after rewriting. This is the conservative, decision-making score.
- **Soft scores (what we log for analysis/reward)**: similarity-weighted precision/recall/F1 that give partial credit when wording differs but meaning aligns. One sentence is softly matched to each claim and vice versa using cosine similarity; we then average and take the harmonic mean:
  - soft_precision = average over review sentences of their best similarity to any claim.
  - soft_recall = average over claims of their best similarity to any review sentence.
  - soft_f1 = harmonic mean of soft_precision and soft_recall.
  These appear in JSON as `soft_precision`, `soft_recall`, `soft_f1` and are used to rank/rerank candidates, not to gate. They explain “soft” quality without changing the hard decision rule.
- **Models**: local 7–8B instruct/coder models (llama3:8b-instruct-q4, deepseek-coder-6.7b-base-q4, qwen2.5-coder-7b) run locally (e.g., Ollama/HF).
- **Prompt styles**: default, concise, evidence, test-heavy (same structure, different tone/length).

## Algorithm (plain)
1) Score seed review → `Rel_seed`.  
2) If `Rel_seed >= τ` (τ≈0.6), keep seed; stop.  
3) Else, rewrite once using chosen prompt style + claims (+ optional diff snippets). Use T≈0.3, 1 sample, short max length.  
4) Score rewrite → `Rel_new`.  
5) Keep the higher-Rel version; log both scores, improvement flag, model, and prompt.

Why single-pass? Cheap, reproducible, and the gate prevents needless edits to already-good reviews.

## Prompt Text (short form shown to the model)
- **default**: Improve to match claims; add missing points, remove irrelevant bits; phrase uncertainty as tests; output only the revision.
- **concise**: Rewrite into 1–3 sentences; mention main change and one concrete check/test; nothing else.
- **evidence**: Add only claim-supported points; if unsure, suggest a test; keep it short.
- **test-heavy**: Emphasize tests/failure modes; add 1–2 concrete tests; avoid obvious restatements.

## Tunable Knobs
- **Threshold τ**: Higher τ rewrites more items; τ≈0.6 balanced trigger rate and quality.
- **Prompt/model**: Concise/default with llama3 or deepseek worked best; test-heavy often bloats length.
- **Sampling**: Single sample, low temperature; raise T only if you want more diversity at higher cost.
- **Scoring strictness**: Same CRScore gates both rewrite triggering and improvement checks.

## Quantitative Evidence
- **Core lift vs. baseline** (`analysis/main_systems.md`): baseline Rel/Con/Comp = 0.116/0.174/0.101 (N=120) → rewrite loop 0.521/0.549/0.511 (~4–5× Rel).
- **Gate matters**: single rewrite without gating hits Rel 0.493; the gate pushes it to 0.521.
- **Best-performing settings** (`analysis/threshold_summary.md`):  
  - Concise + deepseek: Rel 0.789 / Con 0.833 / Comp 0.791; improvement rate 0.858.  
  - Default + llama3: Rel 0.746 / Con 0.767 / Comp 0.777; improvement rate 0.908.  
  - Concise + llama3: Rel 0.701 / Con 0.813 / Comp 0.669; improvement rate 0.875.
- **Per-language snapshots**:  
  - Concise + deepseek (`results/threshold_concise_deepseek-coder_6.7b-base-q4_0_summary.json`): Rel 0.82 (Java) / 0.77 (JS) / 0.78 (Py); overall Rel 0.789, Con 0.833, Comp 0.791.  
  - Concise + llama3 (`results/threshold_concise_llama3_8b-instruct-q4_0_summary.json`): Rel 0.701 overall; Con 0.813.  
  - Concise + qwen (`results/threshold_concise_qwen2.5-coder_7b_summary.json`): Rel 0.646 overall; Con 0.694.  
  Cross-language variance stays modest; no language collapses.
- **Trigger effectiveness** (`analysis/threshold_improvement.md`): Many settings improve ≥70% of gated items; best runs exceed 90%.
- **Language-agnostic baseline check** (`results/rewrite_loop_summary.json`): Rel ≈0.49–0.55 across Java/JS/Python, confirming consistency.

## Practical Guidance
- Keep the gate on; it protects good human reviews and reduces churn.  
- Prefer concise/default prompts; they stay short and align with CRScore.  
- Use small local models (quantized 7–8B) for privacy and speed; one-sample decoding keeps cost and variance low.  
- Fix seeds when possible, but expect mild non-determinism from LLM servers.

## What to Show in the Paper
- Diagram or 4-step schematic: **score → gate → rewrite → re-score/select**.  
- Tables: main systems (baseline vs. single rewrite vs. gated rewrite), plus prompt/model table (concise/default variants).  
- Sensitivity blurb: τ=0.6 is a good balance; higher τ rewrites more but risks over-editing; lower τ saves cost but leaves weak reviews.  
- Cost/latency note: single-pass, one-sample, small models keep batch runtime practical.  
- Threats to validity (pointer: `docs/threats-to-validity.md`): small dataset, pseudo-reference claims, CRScore alignment limits, model/version variability, lack of human eval.

## Reproduction Checklist
1) Gather diff, seed review, and claims.  
2) Choose τ (start at 0.6).  
3) Pick prompt style (concise or default) and model (llama3 8B or deepseek 6.7B).  
4) Decode: T≈0.3, 1 sample, short max length.  
5) Score with CRScore; keep the higher Rel of seed vs. rewrite.  
6) Log Rel/Con/Comp, improvement flag, model, prompt, τ, and seed score for auditing.
