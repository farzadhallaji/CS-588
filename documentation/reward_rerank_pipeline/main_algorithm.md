# Review Reward-Rerank Pipeline

Two-stage system: generate diverse rewrites of a review, then score and pick the best using a relevance-focused reward with penalties.

- **Stage 1 — Generation**: produce several candidate rewrites per item by varying prompt style (default, evidence-grounded, test-heavy, concise) and sampling settings (temperature, sample count). Inputs include the seed review, claims/pseudo-references, and diff/context when available.
- **Stage 2 — Reward scoring**: each candidate receives a reward = relevance to claims minus penalties for unsupported statements, excessive length, and copying input text. A similarity threshold gates what counts as supported evidence; selection temperature adds slight randomness to break ties.
- **Selection output**: for each item, emit the top-scoring review with its component scores, the prompt/sampling tag, and reward configuration.
- **Why it works**: separates exploration (diverse candidates) from discipline (scoring), improving odds of finding a high-quality review without relying on a single prompt or stochastic sample.
- **Beyond selection**: robustness checks perturb inputs to see how stable the chosen outputs are, complementing headline scores.
