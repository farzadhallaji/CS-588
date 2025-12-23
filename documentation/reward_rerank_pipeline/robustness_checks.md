# Robustness Checks

After selecting best reviews, perturbations test whether the system is stable or brittle.

- **Perturbations**: alter inputs (e.g., tweak text around claims or reviews) and re-score to see how relevance and penalties shift.
- **Metrics logged**: score deltas per item, highlighting which outputs degrade under minor changes.
- **Purpose**: confirms the reranker is not overfitting to surface phrasing and that chosen reviews remain strong when inputs vary slightly.
