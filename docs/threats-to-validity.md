# Threats to validity

## Data and labels
- **Single dataset, small test set (120 items)**: results may not generalize to other domains, languages, or larger corpora. The fixed 40-per-language split reduces variance but limits coverage.
- **Pseudo-reference quality**: claims are weak labels; inaccuracies or omissions can skew relevance/comprehensiveness scoring and misguide rewrites.
- **Potential leakage across splits**: deterministic split helps, but if similar or duplicate items exist across train/dev/test, scores may be inflated.

## Metrics and scoring
- **CRScore bias**: reliance on a single embedding-based scorer may not reflect human judgments; models may overfit to CRScore quirks (length, phrasing).
- **Threshold sensitivity**: choices of tau and relevance gates affect triggering and selection; small changes can alter outcomes materially.
- **Penalty weights**: handcrafted weights for unsupported/length/copy penalties may not match human preferences; tuning risk of overfitting to the scorer.

## Modeling and prompts
- **Model variance**: different local models (llama3-style, deepseek-coder, qwen2.5) have varying safety and faithfulness; results depend on the chosen model and version.
- **Sampling randomness**: temperatures and few samples can introduce instability; single-pass runs may miss better candidates or inflate variance.
- **Prompt anchoring**: few-shot examples and prompt wording may bias outputs; examples come from the same domain and may not transfer.

## Experimental design
- **Limited ablation scope**: sweeps cover prompts, temps, samples, and penalties, but not broader factors (e.g., alternative scorers, longer contexts, multi-pass refinement for proposal v1).
- **Robustness coverage**: perturbation checks may not capture real-world distribution shifts (e.g., noisy diffs, multi-file changes, adversarial inputs).
- **No human evaluation here**: all reported numbers are automatic; without human judgments, real perceived quality is uncertain.

## Reproducibility and environment
- **Local runtime differences**: hardware, quantization, and local model builds can change outputs; Ollama/HF-local variations affect reproducibility.
- **Determinism limits**: fixed seeds help, but LLM servers and parallelism can introduce nondeterminism in sampling or scoring.
