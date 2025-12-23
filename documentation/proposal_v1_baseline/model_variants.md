# Model Variants for Proposal v1

Run the baseline with different local LLM backends to see how model choice changes results.

- **Swappable models**: default is a lightweight instruction model; rerun with coder-focused or larger instruction models served locally.
- **What stays fixed**: example pairs, scoring thresholds, prompt wording, and single-pass flow stay the same across models.
- **What to compare**: relevance/conciseness/comprehensiveness scores and the tone/detail of rewrites; watch how temperature or sampling shifts interact with each model.
