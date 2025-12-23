# Generation Variants

Candidate diversity comes from changing prompt style and sampling settings; every variant feeds into the same reward scorer.

- **Prompt sweeps**: run with one prompt style at a time (default, evidence-grounded, concise, test-heavy) to see how direction affects quality.
- **Temperature sweeps**: cooler (~0.2) for precision, warmer (~0.6) for exploration; baseline uses a mid value.
- **Sample-count sweeps**: compare 2 vs 4 vs 8 samples per item; more samples find more good rewrites but cost more.
- **Naming**: each run carries a tag for its prompt/temperature/sample choice so results are easy to trace.

## Prompt text
- **default**
  ```
  You are improving a code review comment.

  Seed review:
  {seed_review}

  Pseudo-references (claims about what should be mentioned):
  {claims}

  Task:
  Rewrite the review so it covers the important claims, stays concise, and does not add new unverifiable claims.
  Return ONLY the rewritten review text.
  ```
- **concise**
  ```
  Rewrite the seed review into at most 3 bullet points.
  Cover as many claims as possible without adding new information.
  No greetings. No filler. Return ONLY the bullets.

  Seed review:
  {seed_review}

  Claims:
  {claims}
  ```
- **evidence_grounded**
  ```
  Rewrite the review, but every statement must be grounded in the provided diff evidence.
  If something is uncertain, phrase it as a question or suggest a verification/test instead of asserting it.

  Seed review:
  {seed_review}

  Claims:
  {claims}

  Diff evidence:
  {diff}

  (Optionally old code context)
  {old_code}

  Uncovered claims to prioritize:
  {uncovered_claims}

  Offending sentences to trim or rewrite:
  {offending_sentences}

  Return ONLY the rewritten review.
  ```
- **test_heavy**
  ```
  Rewrite the review with a verification mindset:
  - If a claim cannot be directly verified from the diff, convert it into a test suggestion or a question.
  - Prefer actionable checks (unit tests, edge cases, logging, invariants).

  Seed review:
  {seed_review}

  Claims:
  {claims}

  Diff:
  {diff}

  Evidence snippets (optional):
  {evidence_snippets}

  Return ONLY the rewritten review.
  ```
