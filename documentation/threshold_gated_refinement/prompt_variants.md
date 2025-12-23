# Prompt Variants

Four prompt styles steer the gated refinement differently; each is used when the relevance gate fires.

- **default**: general improvement toward the claims; add missing points, drop irrelevant text, convert uncertainty into test suggestions.
- **concise**: compress into 1–3 sentences that mention the main change and one concrete check/test; strip everything else.
- **evidence**: only add points directly supported by the claims; if support is missing, suggest a verification/test instead.
- **test-heavy**: emphasize test coverage and failure modes; add 1–2 concrete tests (happy path + edge case) without restating obvious change details.

These can be paired with different local model backends; all use one sample at low temperature to keep outputs stable.

## Exact prompt text
- **default**
  ```
  Improve the review to better match the claims.
  Include missing important points, remove irrelevant content.
  If a claim is uncertain, phrase it as a verification/test request.
  Output only the revised review.
  ```
- **concise**
  ```
  Rewrite the review into 1-3 sentences.
  Must mention the main change and one concrete check/test.
  Remove everything else.
  Output only the revised review.
  ```
- **evidence**
  ```
  Add ONLY points that are directly supported by the claims text.
  If support is missing, do not add the point (suggest a test instead).
  Keep it short and specific.
  Output only the revised review.
  ```
- **test-heavy**
  ```
  Focus on test coverage and failure modes implied by the claims.
  Add 1-2 concrete tests (happy path + edge case).
  Avoid restating obvious change details unless needed.
  Output only the revised review.
  ```
