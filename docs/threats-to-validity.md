# Threats to validity

## Data and labels
- **Narrow dataset (120 items, three languages)**: coverage is limited; results may not transfer to other domains, languages, or larger codebases even with a fixed 40-per-language split.
- **Weak-label pseudo-references**: claims act as noisy references; omissions or hallucinations can distort relevance/comprehensiveness scoring and bias the rewrite policy.
- **Split leakage risk**: deterministic splits reduce variance but cannot rule out near-duplicates across train/dev/test, potentially inflating reported gains.

## Metrics and scoring
- **Single-scorer dependence**: CRScore is embedding-based and may not align with human judgment; systems may overfit quirks such as length or phrasing.
- **Threshold and gate sensitivity**: tau choices and relevance gates materially change triggering rates and selected candidates; small shifts can reverse conclusions.
- **Hand-tuned penalties**: weights for unsupported/length/copy penalties reflect manual preferences and may not match user perception; tuning on the scorer risks overfitting.

## Modeling and prompts
- **Model/version variability**: local models (llama3-style, deepseek-coder, qwen2.5) differ in safety and faithfulness; results depend on the specific build and quantization.
- **Sampling noise**: low sample counts and nonzero temperatures introduce instability; single passes can miss better candidates or exaggerate variance.
- **Prompt anchoring**: few-shot examples are drawn from the same domain; wording and examples may bias outputs and limit transfer.

## Experimental design
- **Incomplete ablations**: sweeps cover prompts, temperatures, sample counts, and penalties but exclude broader factors (alternative scorers, longer contexts, multi-pass refinement for proposal v1).
- **Robustness gaps**: perturbation tests are limited; they may not reflect real-world shifts such as noisy diffs, multi-file changes, or adversarial edits.
- **No human study**: evaluation is fully automatic; without human judgments, user-perceived quality and trustworthiness remain uncertain.

## Reproducibility and environment
- **Local runtime differences**: hardware, quantization settings, and local model builds (Ollama/HF) affect outputs; exact reproduction may require matching the stack.
- **Limited determinism**: fixed seeds help but do not eliminate nondeterminism from LLM servers, parallelism, or sampling order.
