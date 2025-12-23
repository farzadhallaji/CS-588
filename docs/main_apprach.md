# Main Approach: Threshold-Gated Single-Pass Rewrite

This note explains the full method in plain terms. Assume the reader only knows that **CRScore** is the relevance scorer we use for reviews.

## Goal
Upgrade noisy junior code reviews by rewriting only the weak ones, using small local LLMs, while keeping cost and risk low.

## Ingredients
- **Inputs per item**: code diff, seed (junior) review, and a list of claims that describe what a good review should cover.
- **Scorer**: CRScore gives three numbers in `[0,1]` — relevance (Rel), conciseness (Con), and comprehensiveness (Comp). We gate and judge improvements with Rel; Con/Comp are tracked for quality.
- **Models**: local 7–8B instruct/coder models (llama3:8b-instruct-q4, deepseek-coder-6.7b-base-q4, qwen2.5-coder-7b) served locally (e.g., Ollama/HF).
- **Prompt styles**: default, concise, evidence, test-heavy. Styles change tone/length but follow the same structure.

## Step-by-Step Pipeline
1) **Score the seed review** with CRScore → Rel_seed.  
2) **Gate**: if `Rel_seed >= τ` (threshold, typically 0.6), stop and keep the seed review.  
3) **Rewrite once** (only if gated):
   - Build a prompt: style text + current review + claims (and optionally diff snippets).
   - Sample one rewrite with a modest setting (T≈0.3, one sample, short max length).
4) **Re-score the rewrite** → Rel_new.
5) **Select the better review**: keep the rewrite if `Rel_new > Rel_seed`, else keep the seed.
6) **Store outputs**: final review, both scores, whether it improved, and which prompt/model were used.

Why single-pass? It is cheap, easy to reproduce, and the gate avoids touching already-good reviews.

## Prompt Shapes (what the model sees)
- **default**: “Improve the review to better match the claims; add missing key points, remove irrelevant bits; phrase uncertainties as tests; output only the revised review.”
- **concise**: “Rewrite into 1–3 sentences; mention the main change and one concrete check/test; drop everything else.”
- **evidence**: “Add only points supported by the claims; if unsure, suggest a test; keep it short.”
- **test-heavy**: “Emphasize tests and failure modes; add 1–2 concrete tests; avoid restating obvious change details.”

## Key Knobs
- **Threshold τ**: higher τ rewrites more items; τ≈0.6 worked well. Lower τ saves cost but may leave weak reviews untouched.
- **Model choice**: all three local models work; llama3 and deepseek generally scored higher than qwen in these runs.
- **Sampling**: one sample, low temperature. Raising temperature gives diversity but costs more.
- **Scoring strictness**: the same CRScore gate is used to decide when to rewrite and whether it improved.

## Results at a Glance
- **Big lift vs. baseline** (`analysis/main_systems.md`):  
  - Baseline seed reviews: Rel 0.116 / Con 0.174 / Comp 0.101 (N=120).  
  - Threshold-gated rewrite loop: Rel 0.521 / Con 0.549 / Comp 0.511.
- **Single rewrite without gate**: Rel 0.493 / Con 0.505 / Comp 0.504 — good, but the gate still helps.
- **Best prompt/model combos** (`analysis/threshold_summary.md`):  
  - Concise + deepseek: Rel 0.789, Con 0.833, Comp 0.791, improvement rate 0.858.  
  - Default + llama3: Rel 0.746, Con 0.767, Comp 0.777, improvement rate 0.908.  
  - Concise + llama3: Rel 0.701, Con 0.813, Comp 0.669, improvement rate 0.875.
- **Cross-language consistency** (`results/rewrite_loop_summary.json`): Rel ≈0.49–0.55 across Java/JS/Python with similar Con/Comp, showing no language collapses.
- **Improvement rates** (`analysis/threshold_improvement.md`): Many settings improve ≥70% of gated items; best runs exceed 90%.

## Practical Notes
- Keep the gate: it avoids unnecessary rewrites and preserves strong human reviews.
- Prefer concise/default styles for clarity; test-heavy can over-extend length and hurt Rel.
- Use small local models to stay private and cheap; quantized 8B models were enough.
- One-sample decoding simplifies reproducibility; fix seeds where possible, but expect minor non-determinism from LLM servers.

## What to Report in the Paper
- A schematic of the **gate → rewrite → re-score → select** loop.
- The main system table (baseline vs. rewrite loop vs. single rewrite) and one prompt/model table (e.g., concise/default variants).
- A short paragraph on **threshold sensitivity**: τ=0.6 balanced trigger rate and quality; higher τ rewrites more but risks over-editing.
- A brief **cost/latency note**: single-pass, one-sample, small local models keep runtime practical for batch processing.
- **Threats to validity**: small dataset, pseudo-reference claims, CRScore alignment limits, local model variance, and lack of human evaluation (see `docs/threats-to-validity.md` for details).

## Reproduction Checklist
- Inputs: diff, seed review, claims.  
- Set τ (e.g., 0.6).  
- Pick prompt style (concise or default recommended) and model (llama3 8B or deepseek 6.7B).  
- Decode with T≈0.3, 1 sample, short max length.  
- Score with CRScore; keep the better of seed vs. rewrite.  
- Log scores, improvement flag, and chosen variant.
