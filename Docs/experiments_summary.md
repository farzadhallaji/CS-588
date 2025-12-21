Experiments and Ablations (Summary)
===================================

Scope and Data
--------------
- Dataset: 300 CRScore phase 1 code changes (100 each in Java/JS/Python), with seed human reviews and claim lists.
- Metric: CRScore Conciseness (Con), Comprehensiveness (Comp), Relevance (Rel = F1) at τ = 0.7314.

Systems Run
-----------
Offline/template editors (no LLM):
- Loop (edit, selection on), single edit (k1), single rewrite, rewrite loop (iterative full rewrites).
- Ablations: no selection (random pick), no evidence (guardrail off).

Threshold-gated local LLM refinement:
- Trigger: only refine if seed Rel < 0.6.
- Models: DeepSeek-Coder 6.7B, Llama 3 8B, Qwen 7B (local).
- Prompts: default (general improve), concise (1–3 sentences + one test), evidence (add only supported points), test-heavy (focus on tests/edge cases).

Key Results (Rel headline, test split)
--------------------------------------
- Seed humans: ~0.12 Rel (baseline).
- Offline rewrite loop: ~0.52 Rel (best non-LLM; single rewrite ~0.49; edit loops ~0.15–0.16).
- Threshold-gated DeepSeek 6.7B:
  - default ≈ 0.79 Rel; concise ≈ 0.79 Rel (higher Con, slightly lower Comp).
- Threshold-gated Llama 3 8B:
  - default ≈ 0.75 Rel.
- Evidence/test-heavy prompts: generally lower Rel; evidence is safest but slightly weaker.

Ablation Takeaways
------------------
- Iteration helps when rewriting (rewrite loop > single rewrite).
- Minimal edits stay weak; disabling selection or evidence guardrails does not rescue them.
- Evidence guardrails increase precision-like behavior; removing them risks unsupported additions.
- Threshold gating (only refining low-Rel seeds) focuses LLM effort where it matters.

Outputs and Scoring
-------------------
- Results stored under `results/` (JSONL per system) with summaries; plots under `analysis/plots/`.
- Combined scoring script evaluates all CSV columns plus threshold DeepSeek runs and rewrite loop; outputs JSON/CSV and Rel-by-system plots.

What is Supported vs. Not Claimed
---------------------------------
- Supported: Automatic CRScore gains on phase 1 test split; relative ranking of systems and ablations.
- Not claimed: Human-judged quality/faithfulness, cross-domain generality, or cost/latency advantages (no human eval or robustness study yet).
