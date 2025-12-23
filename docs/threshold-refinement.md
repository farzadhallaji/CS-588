# Threshold-gated refinement

Single-pass local rewrite that only triggers when the starting review scores low on relevance.

## How it works (no code needed)
- **Input**: for each item, you have a code diff, a junior review, and “claims” that describe what a good review should cover.
- **Score first**: measure how relevant the junior review is to the claims. Call this number Rel.
- **Gate**: if Rel is above a chosen threshold, keep the review as-is. If Rel is below the threshold, try to improve it once with an LLM.
- **Refine** (only when gated):
  - Build a prompt from a chosen style, the current review, and the claims.
  - Ask a local LLM for one rewritten review. Typical settings: temperature around 0.3, one sample, a modest max length. You can use different local models (e.g., llama3-style, deepseek-coder, qwen2.5-coder) or HF-local backends.
  - Re-score the rewritten review; mark it improved if its Rel exceeds the seed Rel.
- **Output**: for each item, keep the original and the best (possibly unchanged) review, plus their scores and flags indicating whether it improved.

## Prompt styles you can pick
- **default**: Make the review better aligned to the claims; add missing key points, remove irrelevant bits; if unsure, phrase as a test/verification.
- **concise**: Rewrite into 1–3 sentences; must mention the main change and one concrete check/test; drop everything else.
- **evidence**: Add only points directly supported by the claims; if support is missing, suggest a test; keep it short.
- **test-heavy**: Emphasize tests and failure modes; add 1–2 concrete tests (happy path + edge case); avoid restating obvious change details unless needed.

Prompt shape (conceptual):
```
<style text>

Current review:
{review}

Claims:
- {claim1}
- {claim2}
...

Revised review:
```

Exact style texts (what the model sees before the review/claims block):
- **default**  
  Improve the review to better match the claims.  
  Include missing important points, remove irrelevant content.  
  If a claim is uncertain, phrase it as a verification/test request.  
  Output only the revised review.

- **concise**  
  Rewrite the review into 1-3 sentences.  
  Must mention the main change and one concrete check/test.  
  Remove everything else.  
  Output only the revised review.

- **evidence**  
  Add ONLY points that are directly supported by the claims text.  
  If support is missing, do not add the point (suggest a test instead).  
  Keep it short and specific.  
  Output only the revised review.

- **test-heavy**  
  Focus on test coverage and failure modes implied by the claims.  
  Add 1-2 concrete tests (happy path + edge case).  
  Avoid restating obvious change details unless needed.  
  Output only the revised review.

## Knobs and ablations
- **Threshold**: how low Rel must be to trigger refinement (e.g., 0.6 by default). Lower thresholds fire less often; higher thresholds rewrite more items.
- **Prompt style**: choose among default, concise, evidence, test-heavy to steer the rewrite.
- **Model**: swap among local LLMs (e.g., llama3-style, deepseek-coder 6.7B, qwen2.5-coder 7B) to see model effects.
- **Sampling**: adjust temperature, top-p, max tokens, and stick to one sample to keep it cheap; you can raise temperature for more variation.
- **Scoring strictness**: adjust the similarity threshold used to decide Rel, which controls how often the gate triggers and how improvements are judged.

## What this experiment shows
- How much a simple “rewrite only when relevance is low” gate can lift relevance, conciseness, and comprehensiveness.
- Which prompt style works best under the same gate.
- How sensitive results are to the chosen local model and to the gate threshold.
