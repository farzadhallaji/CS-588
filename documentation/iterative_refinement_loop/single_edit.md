# Single-Edit Variant (k1)

One-pass version of the iterative loop that stresses high-quality edits without multiple iterations.

- **What changes**: only one revision attempt; prompt text urges the model to deliver its best single rewrite while trimming low-signal sentences.
- **Why run it**: isolates the value of multi-step refinement by comparing a single decisive edit against the full loop.
- **Behavior to note**: still filters candidates for evidence support, conciseness, and minimal change, then picks the best valid option by relevance-focused objective.
