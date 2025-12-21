Long-form experiment guide (plain language)
===========================================

Audience
--------
- A reader new to CRScore and this repository.
- No code knowledge assumed; we describe goals, methods, results, and limits in plain language.

What we are trying to do
------------------------
Given a code change and an initial human review comment, we want to produce a better review: concise, complete, and grounded in the actual change. Quality is measured with CRScore:
- **Conciseness (Con):** precision-like; how many review sentences align with the target claims.
- **Comprehensiveness (Comp):** recall-like; how many target claims are covered.
- **Relevance (Rel):** F1 of Con and Comp; the main headline metric.
We use a similarity threshold τ = 0.7314 to decide whether a review sentence matches a claim.

Data used
---------
- 300 code changes (100 each in Java, JavaScript, Python) from the CRScore phase 1 human study.
- Each change includes: the diff, a human “seed” review, several model baseline reviews, and a list of claims (facts the review should cover).
- All experiments here use only this phase 1 data (no phase 2 labels for tuning).

Systems evaluated
-----------------
We evaluate two families of systems.

1) Offline template-based editors (iterative loops)
   - Start from the human seed review.
   - Score it with CRScore to find missing or low-similarity points.
   - Generate 1–2 candidate edits per iteration using simple templates (no LLM), either minimal edits or full rewrites.
   - Guardrails: avoid adding unsupported content (evidence checks) and avoid drops in conciseness.
   - Select the best candidate per iteration using CRScore (unless selection is disabled).
   - Iterate up to 3 times (or 1 time for single-pass ablations).
   - Variants:
     - Loop (edit): minimal edits, selection on.
     - Single edit (k1): one-pass minimal edit.
     - Single rewrite: one-pass full rewrite (no minimal-edit constraint).
     - Rewrite loop: iterative full rewrites (best offline variant).
     - No selection: same as loop but pick a candidate at random (tests selection value).
     - No evidence: same as loop but without evidence guardrails (tests grounding effect).

2) Threshold-gated refinement with small local models (LLMs)
   - Score the human seed; if Rel ≥ 0.6, keep it.
   - If Rel < 0.6, ask a local LLM once with a guiding prompt.
   - Prompts (paired with small local models such as DeepSeek-Coder 6.7B or Llama 3 8B):
     - default: Improve to match claims; add missing points; remove irrelevant content; if unsure, ask for a test.
     - concise: Rewrite to 1–3 sentences; mention main change and one concrete test.
     - evidence: Add only claim-supported points; otherwise suggest a test.
     - test-heavy: Emphasize tests and edge cases.
   - These runs deliver the highest automatic scores; DeepSeek-Coder 6.7B with default or concise is the top performer.

What the main run does
----------------------
- Runs the offline template systems and ablations on the test split.
- Runs the threshold-gated refinements across prompt/model pairs.
- Writes per-system outputs to results/ and aggregates to analysis/.
- Reruns are idempotent: existing outputs are reused unless removed.

What the combined scoring script does
-------------------------------------
- Reads the curated CSV containing seed reviews, baseline model reviews, and a custom “enhanced” review column.
- Adds the best reviews from the threshold-gated DeepSeek runs (default/concise) and the offline rewrite loop.
- Scores every system with CRScore (Con/Comp/Rel), overall and by language.
- Reports deltas vs the human seed and vs the custom enhanced review.
- Produces JSON and CSV summaries, plus plots of Relevance overall and by language.

Results (headline numbers; Rel is primary)
------------------------------------------
- Threshold-gated DeepSeek 6.7B (default prompt): Rel ≈ 0.79 (high Con and Comp).
- Threshold-gated DeepSeek 6.7B (concise prompt): Rel ≈ 0.79 (slightly higher Con, slightly lower Comp).
- Threshold-gated Llama 3 8B (default): Rel ≈ 0.75.
- Offline rewrite loop: Rel ≈ 0.52 (best non-LLM variant; iteration beats single rewrite at ~0.49).
- Minimal-edit loops and ablations: much weaker (Rel ≈ 0.15–0.49).
- Human seed reviews: low Rel (~0.12), leaving large headroom.

Conclusions (what we can responsibly claim)
-------------------------------------------
- Small local LLMs with threshold gating and concise/default prompts yield the strongest automatic scores (Rel ~0.79), substantially above the human seeds and above all template-based variants.
- Iterative full rewrites (offline) outperform single rewrites and any minimal-edit loop, showing iteration helps when rewriting capacity is present.
- Removing selection or evidence guardrails does not improve the weak edit variants; quality is limited by the edit generator, not the guardrails.
- Evidence-aware prompts/guardrails increase precision-like behavior; the “evidence” prompt is safest but slightly weaker on Rel.

Limits (what we cannot claim without further work)
--------------------------------------------------
- No human evaluation is reported here (phase 2 labels are not used); all numbers are automatic CRScore. We cannot claim human-perceived quality gains without new human ratings.
- No hallucination/faithfulness audit beyond evidence guardrails and conciseness; we cannot claim absence of unsupported statements.
- No cost or latency analysis; we cannot claim the cheapest/fastest configuration.
- Dataset scope is limited to CRScore phase 1 (Java/JS/Python); we cannot claim cross-language or cross-domain generality beyond this set.

How to read the outputs (no code needed)
----------------------------------------
- JSON: overall and per-language Con/Comp/Rel for each system, plus deltas vs baseline (seed) and the enhanced reference.
- CSV: flat table for quick comparison or spreadsheets.
- Plots: bar charts of overall Rel by system, and grouped bars of Rel by system and language.

If you want to extend or reproduce
----------------------------------
- Keep τ=0.7314 and the same claims as targets; use the same three-language test split.
- For best scores: threshold-gated DeepSeek (default/concise). For safer/grounded content: use the evidence prompt. For non-LLM: use the rewrite loop.
- To compare new systems: align on the same `index` ids, add your reviews to the CSV or JSONL, and rescore with the combined scoring script.
