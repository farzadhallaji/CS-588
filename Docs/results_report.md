# Results and Analysis (Paper Reference)

This document consolidates the experiment design, systems, prompts, and empirical findings based on the current `results/` and `analysis/` artifacts. It is written for a paper/presentation audience (no code details).

## What the task is
We start from human-written code reviews and a list of “claims” (pseudo-references) extracted from the code change. The goal is to improve the reviews so they cover important claims (recall), avoid irrelevant content (precision), and stay concise.

## What the dataset contains
- Source: CRScore human study phase 1.
- Size: 300 total instances; test split used here: 120 (40 per language: Java, JS, Python).
- Per instance fields:
  - `index`: unique id within the dataset (integer).
  - `lang`: programming language (`java`, `js`, or `py`).
  - `id`: original project-specific identifier.
  - `msg`: the “seed” human review — the raw review comment written by a human annotator before any model editing.
  - `patch`: the code diff for the change.
  - `oldf`: the original file content (used for evidence retrieval).
  - `claims`: the pseudo-references (target facts the review should cover). Format varies:
      - A string (claim text), or
      - A pair/list where the second element is the claim text (e.g., `[claim_id, "claim text"]`).

## How quality is measured
- **CRScore metrics**: Conciseness (Con, precision-like), Comprehensiveness (Comp, recall-like), and Relevance (Rel = F1 of Con/Comp). Higher is better.
- **Similarity threshold τ**: 0.7314 (from the CRScore paper); a claim is “covered” if its best sentence match exceeds τ.
- **Evidence guardrail τ_evidence**: 0.35; added sentences must be semantically supported by evidence above this threshold.

## Method overview (plain language)
1) **Offline/template editor**: A lightweight, non-LLM template edits the review over 1–3 iterations, guided by CRScore to pick the best candidate. Variants differ in how many iterations, whether they rewrite vs minimally edit, and whether they enforce evidence or selection.
2) **Few-shot baseline (proposal v1)**: A small local LLM rewrites using a fixed prompt plus mined “bad→good” pairs.
3) **Threshold-gated refinement**: Score the seed review; if its Rel < 0.6, ask a local LLM (Ollama) to refine it using a chosen prompt. This is where most quality gains come from in our runs.

## What the system names mean (offline/template editor)
- **baseline**: the original human review, unchanged.
- **single_edit (k1)**: one-shot minimal edit using the template editor.
- **loop**: multi-iteration minimal edits; at each step we generate candidates and pick the best via CRScore.
- **single_rewrite**: one-shot full rewrite (no minimal-edit constraint).
- **rewrite_loop**: multi-iteration full rewrites (best offline variant).
- **no_selection**: same as loop, but picks a candidate at random (tests whether CRScore-based selection matters).
- **no_evidence**: same as loop, but disables the evidence guardrail (tests grounding).

## What the prompt names mean (threshold-gated LLM refinement)
- **default**: general improvement; add missing key points, drop irrelevant content; suggest tests when uncertain.
- **concise**: compress to 1–3 sentences; must mention the main change and one concrete test.
- **evidence**: only add points directly supported by the claims; otherwise suggest a test.
- **test-heavy**: focus on test coverage and failure modes (happy path + edge case).

### Exact prompt texts (threshold refinement)
- **default**  
  ```
  Improve the review to better match the claims.
  Include missing important points, remove irrelevant content.
  If a claim is uncertain, phrase it as a verification/test request.
  Output only the revised review.
  ```
- **concise**  
  ```
  Rewrite the review into 1-3 sentences.
  Must mention the main change and one concrete check/test.
  Remove everything else.
  Output only the revised review.
  ```
- **evidence**  
  ```
  Add ONLY points that are directly supported by the claims text.
  If support is missing, do not add the point (suggest a test instead).
  Keep it short and specific.
  Output only the revised review.
  ```
- **test-heavy**  
  ```
  Focus on test coverage and failure modes implied by the claims.
  Add 1-2 concrete tests (happy path + edge case).
  Avoid restating obvious change details unless needed.
  Output only the revised review.
  ```

### How each experiment runs (algorithm sketch)
- **Offline/template loop (all ablations above)**  
  1) Score the current review with CRScore to find uncovered claims (below τ) and low-similarity sentences.  
  2) (For evidence-enabled variants) retrieve evidence snippets from the diff/old file; filter new sentences that lack semantic support (τ_evidence).  
  3) Generate 2 template-based candidates (minimal edit or rewrite, depending on the variant).  
  4) Score candidates; enforce constraints (length, precision drop, max change ratio).  
  5) Select the best by objective (CRScore Rel penalized by length/copy) unless in `no_selection` (random). Repeat up to K iterations (K=3 unless K=1 in single_edit).
- **Proposal v1 (few-shot baseline)**  
  - Use mined “bad→good” pairs from the dev split.  
  - Prompt a local LLM once with the current review and few-shots to produce an improved review (no iteration, no threshold gating).
- **Threshold-gated refinement (Ollama models, prompts above)**  
  1) Score the seed review; if Rel ≥ 0.6, keep it.  
  2) If Rel < 0.6, build a prompt with the chosen variant and the claims; call the local LLM once.  
  3) Keep the refined review; record both seed and final scores and whether improvement occurred.

## Data, Metrics, and Thresholds
- **Dataset**: CRScore human study phase 1 (test split size: 120 instances; 40 per language: Java, JS, Python).
- **Automatic metric**: CRScore Conciseness (Con), Comprehensiveness (Comp), and F1 Relevance (Rel). Default τ = 0.7314 (per CRScore paper).
- **Threshold-gated refinement**: Trigger condition Rel < 0.6 on the seed review; τ for scoring remains 0.7314.
- **Evidence guardrail**: τ_evidence = 0.35 for semantic support checks.

## Systems and Ablations (offline/template editor)
Outputs and summaries live under `results/` and aggregated tables in `analysis/main_systems.md`/`.csv`.
- **baseline**: unmodified human seed review (Rel=0.116).
- **loop**: iterative edit with CRScore selection (Rel=0.158).
- **single_edit (k1)**: one-pass edit, same inputs (Rel=0.154).
- **single_rewrite**: one-pass full rewrite (Rel=0.493).
- **rewrite_loop**: iterative full rewrites (Rel=0.521) — best offline variant.
- **no_selection**: loop without CRScore-based selection (Rel=0.158).
- **no_evidence**: loop without evidence guardrail (Rel=0.158).

**Key takeaways**
- Iterative rewrites dominate other template-based variants (Rel 0.521 vs ~0.15–0.49).
- Removing selection or evidence did not improve the weak edit variants; precision/recall remain low in those modes.
- The baseline Rel (0.116) shows substantial headroom; any meaningful edit/rewrite should be compared against it.

## Proposal v1 Few-Shot Baseline (Ollama sweep)
Summaries in `analysis/proposal_v1.md`/`.csv`; per-model outputs in `results/proposal_v1_<model>_*.jsonl`.
- **qwen2.5-coder_7b**: Rel=0.308 (Con=0.386, Comp=0.306).
- **llama3_8b-instruct-q4_0**: Rel=0.303 (Con=0.363, Comp=0.311).
- **deepseek-coder_6.7b-base-q4_0**: Rel=0.094 (Con=0.089, Comp=0.142).

**Takeaway**: Among small local models, Qwen 7b and Llama3 8b produce comparable mid-tier quality; DeepSeek 6.7b underperforms on this setup.

## Threshold-Gated Refinement (Ollama × prompts)
Summaries in `analysis/threshold_summary.md`/`.csv`; improvement rates in `analysis/threshold_improvement.md`/`.csv`; per-run JSONL in `results/threshold_<prompt>_<model>.jsonl`. Prompts are:
- **default**: general improvement.
- **concise**: 1–3 sentences, must mention main change + one concrete test.
- **evidence**: add only claims-supported points; otherwise suggest a test.
- **test-heavy**: emphasize test coverage and edge cases.

**Overall Rel (higher is better; N=120)**
- **default (deepseek-coder_6.7b-base-q4_0)**: Rel 0.788, Con 0.793, Comp 0.838.
- **concise (deepseek-coder_6.7b-base-q4_0)**: Rel 0.789, Con 0.833, Comp 0.791.
- **default (llama3_8b-instruct-q4_0)**: Rel 0.746, Con 0.767, Comp 0.777.
- **concise (llama3_8b-instruct-q4_0)**: Rel 0.701, Con 0.813, Comp 0.669.
- **default (qwen2.5-coder_7b)**: Rel 0.676, Con 0.692, Comp 0.703.
- **concise (qwen2.5-coder_7b)**: Rel 0.646, Con 0.694, Comp 0.654.
- Evidence/test-heavy prompts are generally weaker on Rel, except test-heavy qwen2.5-coder_7b (Rel 0.668; strong Comp 0.791).

**Improvement rates (fraction of cases improved when triggered; from JSONL)**
- Range: ~0.57 (evidence, qwen2.5) up to ~0.91 (default, llama3).
- Average Rel gains range ~0.32–0.67 across prompt/model pairs.

**Takeaways**
- Threshold gating with local LLMs yields the strongest automatic scores in the study, substantially above offline rewrites.
- For Rel, the best-performing pairs are default/concise with DeepSeek 6.7b, followed by default Llama3 8b. Qwen 7b is competitive but lower.
- Prompt choice matters: “default” and “concise” dominate; “evidence” is safest but less effective; “test-heavy” varies by model (good recall for Qwen).

## Plots (for paper slides/figures)
Generated to `analysis/plots/`:
- `main_systems_rel.png`: bar chart of Rel for core systems/ablations.
- `proposal_v1_rel.png`: bar chart of Rel for proposal v1 models.
- `threshold_rel.png`: grouped bar chart of Rel by prompt × model.

## Recommended Paper Storyline
- **Claim 1 (iteration vs rewrite)**: Iterative rewrite (Rel 0.521) vs baseline (0.116) and single-pass rewrite (0.493) shows iteration helps beyond one-shot rewrites.
- **Claim 2 (metric-guided selection & evidence)**: Weak edit settings show no gains without stronger editing; selection/evidence do not harm, but the quality bottleneck is the template editor—motivate LLM thresholds.
- **Claim 3 (local LLM refinement)**: Threshold-gated refinement with small local models achieves Rel up to ~0.79, far surpassing offline loops; demonstrates a practical, cost-conscious path.
- **Claim 4 (prompt sensitivity)**: Default/concise prompts consistently outperform evidence/test-heavy in Rel; prompt choice is a key lever for local models.
- **Claim 5 (model choice)**: Among small local models, DeepSeek 6.7b and Llama3 8b lead in threshold refinement; Qwen 7b is competitive but behind on Rel; DeepSeek lags in proposal v1 few-shot setting.

## How to Cite Results in the Paper
Use the summarized numbers above or extract from:
- Core systems: `analysis/main_systems.md`.
- Proposal v1 sweep: `analysis/proposal_v1.md`.
- Threshold sweeps: `analysis/threshold_summary.md` and `analysis/threshold_improvement.md`.
- Figures: `analysis/plots/`.

## Notes and Caveats
- Human phase2 evaluation was dropped (no overlapping rated ids for the systems/split). All reported metrics are CRScore-based.
- Threshold runs include both seed and refined scores in JSONL; Rel improvements are computed only on triggered cases (Rel < 0.6 seeds).
- All summaries use the test split only (120 instances). If you rerun with different splits or limits, regenerate analysis tables/plots. 

## Safeguards and Limitations (hallucination/faithfulness)
- **What we enforce automatically**: evidence guardrail (τ_evidence=0.35) blocks additions that are not semantically supported by diff/old-file snippets; the “evidence” prompt variant restricts additions to claim-supported points; threshold gating only rewrites low-quality seeds (Rel < 0.6) to avoid gratuitous edits.
- **What we report**: CRScore metrics (Con/Comp/Rel) and improvement rates. High conciseness and targeted improvements suggest edits are not adding random content, but this is still an automated proxy.
- **What is missing**: no human faithfulness/hallucination study for these runs; phase2 ratings do not overlap the current systems/split.
- **Defensible phrasing**: “We use evidence guardrails and report precision-like (Con) and F1 (Rel) metrics; we did not run a dedicated human hallucination audit on these outputs.”
- **Quick human spot-check (recommended if time permits)**:
  1) Sample ~30 threshold-refined outputs across top prompt/model pairs (e.g., default/concise DeepSeek + Llama3).
  2) Ask a rater to label each sentence as supported/unsupported by the claims/diff; mark any hallucinations.
  3) Report the fraction of unsupported statements; if none or low, cite as a sanity check.
