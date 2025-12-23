# Iterative Refinement Loop

Multi-pass editor that repeatedly improves a review using reference claims, evidence from the diff, and an automatic relevance scorer.

- **Inputs**: junior review, pseudo-reference claims describing what a good review should cover, diff/old code for evidence.
- **Loop flow** (3 iterations by default):
  1. Score the current review to spot uncovered claims and weak sentences.
  2. Pull evidence snippets from the diff/old code to back any new statements.
  3. Build a prompt listing the current review, uncovered claims, offending sentences, and evidence; ask a local editor for a few candidate rewrites.
  4. Score each candidate; drop ones that are too long, unsupported by evidence, or change too much when in “edit” mode.
  5. Pick the best remaining candidate using relevance with small length/copy penalties.
  6. Stop early if relevance stops improving.
- **Output**: best review, its scores, and per-iteration history (uncovered claims, candidates, and the chosen revision).
- **Why it works**: separates critique from generation, keeps edits minimal unless rewrite mode is on, and blocks hallucinated additions by checking evidence support.

## Prompt text (loop family)
- **System guardrails**: “You are a senior code reviewer… do not invent facts… prefer short, high-signal feedback… output only the revised review.”
- **Loop / k1**: lists the current review, uncovered pseudo-references, offending sentences, and evidence snippets; instructions say cover uncovered items, trim offending text, avoid unsupported additions, and stay within a length budget.
- **Rewrite mode**: same structure but says “rewrite the review from scratch for maximum quality,” still grounding in uncovered pseudo-references and staying concise.
- **No-evidence mode**: same layout without evidence snippets; instructions ban unverifiable assertions and encourage test suggestions when unsure.
- **No-selection ablation**: same prompt as loop, but selection ignores the objective (random/shortest among valid) to show the selector’s effect.

### Example instruction text
```
You are a senior code reviewer. Improve the given review comment.
Constraints:
* Be specific to the diff/claims provided.
* Do NOT invent facts, files, or behaviors not supported by the inputs.
* Prefer short, high-signal feedback (correctness, edge cases, tests, risks).
* Avoid generic advice and style nits unless they directly affect correctness.
Output ONLY the revised review text (no explanations, no bullets unless the review itself uses bullets).
```
Loop prompt suffix (loop/k1):
```
Task: minimally edit the current review.
Goals (in priority order):
1. Cover the uncovered pseudo-references using the evidence snippets when available.
2. Remove or rewrite the offending low-similarity sentences.
3. Keep or improve precision: do not add unrelated points.
Rules:
* If evidence does not support a new statement, do not add it.
* If you are unsure, suggest a test instead of asserting behavior.
Output only the revised review.
```
