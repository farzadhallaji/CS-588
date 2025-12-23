# No-Evidence Variant

Ablation that turns off evidence retrieval to see how the loop behaves when grounding is unavailable.

- **What changes**: prompts exclude retrieved evidence and ask the editor to rely only on pseudo-reference claims and the current review.
- **Why run it**: gauges robustness when diffs or evidence snippets cannot be supplied, and highlights the value of grounding.
- **Behavior to note**: stricter instructions to avoid unverifiable assertions; selection still prioritizes relevance while penalizing length and copying.
