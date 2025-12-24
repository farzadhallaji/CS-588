# Experimental Conclusions

- The threshold-gated, single-pass rewrite (τ≈0.6) beats all other setups: baseline relevance 0.116 → 0.521 with gating, and up to ~0.79 with the best prompt/model pair. The gate matters—ungated single rewrite tops out at 0.493 Rel (`analysis/main_systems.md`).
- Best-performing settings (`analysis/threshold_summary.md`): **concise + deepseek 6.7B** (Rel 0.789, Con 0.833, Comp 0.791, 86% improved items) and **default + deepseek 6.7B** (Rel 0.788, Con 0.793, Comp 0.838, 87.5% improved). Choose concise for brevity, default for slightly broader coverage.
- Strong runner-up: **default + llama3 8B** (Rel 0.746, Con 0.767, Comp 0.777, 91% improved). **Concise + llama3** is shorter but trails on Comp (Rel 0.701, Con 0.813, Comp 0.669).
- **Qwen2.5 7B** trails both deepseek and llama3 across prompts (best Rel 0.676–0.646; Con 0.694; Comp 0.703–0.654), so use only if the preferred models are unavailable.
- Prompt style matters: **concise** and **default** dominate; **evidence** and **test-heavy** variants drop Rel/Comp despite decent improvement rates.
- Cross-language stability is good: with concise + deepseek, Rel stays ≥0.77 for Java/JS/Python, so the gains are not tied to a single language set.

**Recommendation:** Keep the τ≈0.6 gate on to protect good seed reviews and avoid wasted rewrites. Default to concise or default prompts with deepseek 6.7B; fall back to llama3 8B if needed. Avoid evidence/test-heavy unless you need their specific tone despite lower quality.
