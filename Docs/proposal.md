Proposal: Faithful, Evidence-Aware Optimization of Code Reviews
===============================================================

Context and Motivation
----------------------
- We start from the CRScore phase 1 dataset (300 Java/JS/Python diffs) with human seed reviews and claim lists.
- Current metric-chasing baselines (including CRScore++/RLHF) focus on improving automatic scores but lack strong evidence that they reduce unsupported content or generalize.
- Our existing systems show high automatic gains with small local models and template rewrites, but we have no human validation or faithfulness guarantees.

Problem
-------
Improve code reviews so they are concise, comprehensive, and faithful to the code changes and claims, while demonstrating generalization beyond the in-domain CRScore setting and validating gains with humans.

Key Ideas (Technical Contributions)
-----------------------------------
1) **Faithfulness-first training objective**
   - Make CRScore differentiable with soft precision/recall surrogates.
   - Add an evidence-support term: every added sentence must be semantically supported by diff/context; penalize unsupported additions.
   - Joint objective = soft Rel gain + evidence support − length/verbosity penalty.

2) **Alignment-aware supervision**
   - Produce sentence-to-claim alignment probabilities; surface them at inference time for transparency.
   - Use alignment confidence to downweight low-support sentences during training (self-regularization).

3) **Robustness to noisy claims**
   - Train with simulated claim noise (dropped/perturbed claims) and adversarial diffs; measure how the model degrades vs baselines.
   - Add a “humility” behavior: when support is weak, emit test requests instead of hallucinated claims.

Systems to Compare
------------------
- Human seed reviews (baseline).
- Template editors: loop, single edit, single rewrite, rewrite loop, and their ablations (selection/evidence off).
- Threshold-gated small models (DeepSeek 6.7B, Llama3 8B) with default/concise/evidence/test-heavy prompts.
- **New:** Differentiable, evidence-aware objective fine-tuning of a small model (same model family as above) with alignment outputs.

Data and Splits
---------------
- Primary: CRScore phase 1 (Java/JS/Python), dev/test splits as in current runs.
- Robustness: an out-of-domain or cross-language slice (e.g., new repos or code-smell sets) to test generalization.
- Human eval set: a held-out subset (e.g., 100 instances) for human rating.

Training and Objectives
-----------------------
- Soft Con/Comp: replace hard threshold with softmax over sentence–claim similarities; derive a smooth F1 surrogate.
- Evidence support: max similarity of each review sentence to diff/context; penalize sentences below a support margin.
- Humility: when claim coverage is uncertain, incentivize test suggestions over speculative statements.
- Loss = soft Rel + evidence term − verbosity penalty; explore joint or alternating updates with the encoder.

Evaluation Plan
---------------
Automatic (all systems):
- CRScore (Con/Comp/Rel) at τ=0.7314 on test split.
- Faithfulness proxy: fraction of sentences with evidence similarity above a threshold.
- Robustness: performance under claim noise and out-of-domain diffs.

Human (selected systems):
- Relevance/faithfulness ratings on held-out examples (blind, multi-rater).
- Unsupported statement rate (sentence-level).
- Preference study between top systems (ours vs threshold-gated baseline vs rewrite loop).

Ablations
---------
- With/without evidence term; with/without humility behavior.
- Hard vs soft (differentiable) objective.
- Joint vs frozen encoder.
- Prompt variants (default/concise/evidence) applied after fine-tuning.

Artifacts
---------
- Reproducible scripts to train and evaluate the differentiable objective.
- JSON/CSV summaries and plots (overall and per language).
- Alignment visualizations (sentence↔claim support) for transparency.

Risks and Mitigations
---------------------
- **Overfitting to CRScore:** mitigate with robustness sets and human eval.
- **Faithfulness not actually improved:** run human audits; tune evidence penalties; report unsupported rates.
- **Optimization instability:** start with frozen encoder; anneal softness/temperature; monitor gradient norms.

Success Criteria
----------------
- Automatic: Rel improves over threshold-gated baselines **and** faithfulness proxy improves or stays flat.
- Human: higher relevance/faithfulness scores and lower unsupported rates vs baselines.
- Robustness: graceful degradation under noisy claims; no collapse in out-of-domain evals.

Timeline (indicative)
---------------------
- Week 1–2: Implement soft CRScore + evidence loss; pilot on dev split.
- Week 3: Robustness setup (claim noise, out-of-domain); stabilize training.
- Week 4: Human study design and pilot (instruction sheet, sampling).
- Week 5: Full runs + human ratings; collect alignment visualizations.
- Week 6: Analysis, plots, paper writing; iterate on ablations if needed.
