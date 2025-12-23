# Proposal v1 baseline

Single-pass few-shot rewrite: improve a junior review using examples and feedback from an automatic scorer.

## How it works (no code needed)
- **Input**: each item has a code change (diff), a junior review, and “claims” describing what a good review should cover.
- **Examples to learn from**: a small set of paired examples showing a low-quality review and its improved version.
- **Score the seed**: an automatic scorer measures the junior review on three dimensions—Relevance, Conciseness, Comprehensiveness—each from 0 to 1. This is the “seed score.”
- **Build the prompt**: stitch together the example pairs, the seed’s scores, the code diff, and the junior review. The instructions tell the model to use the scores as guidance: add missing points if comprehensiveness is low, delete fluff if conciseness is low, align to the change if relevance is low; do not invent details; prefer 1–4 strong sentences.
- **Rewrite once**: ask a local LLM (e.g., an 8B instruction-tuned model) to produce one improved review. Low temperature (~0.2) keeps it focused; if it outputs nothing, keep the original.
- **Score again**: run the same automatic scorer on the improved review to see gains.
- **Save**: keep the original and improved reviews, their scores, and the method tag.

## Prompt shape (conceptual)
```
Example (low quality):
<bad_example_1>
Example (improved):
<good_example_1>

... more pairs ...

A quality assessment has been performed (0..1):
Comprehensiveness, Relevance, Conciseness.
Use lower dimensions to guide improvement.

Scores: {seed scores}

Code change (diff):
{diff}

Junior review:
{review}

Use the examples to transform the junior review into a higher-quality review.
Use the score dimensions to guide edits:
- If Comp is low: add missing key points from the change.
- If Con is low: delete vague/unrelated sentences.
- If Rel is low: align with the actual change and implications.
Rules:
- Do not invent details not supported by the diff or implied evidence.
- Prefer 1-4 high-signal sentences.

Improved review:
```

## Exact prompt text (what the model sees)
```
Example (low quality):
<bad_example_1>
Example (improved):
<good_example_1>

... more pairs ...

A quality assessment has been performed (0..1):
Comprehensiveness, Relevance, Conciseness.
Use lower dimensions to guide improvement.

Scores: {seed scores}

Code change (diff):
{diff}

Junior review:
{review}

Use the examples to transform the junior review into a higher-quality review.
Use the score dimensions to guide edits:
- If Comp is low: add missing key points from the change.
- If Con is low: delete vague/unrelated sentences.
- If Rel is low: align with the actual change and implications.
Rules:
- Do not invent details not supported by the diff or implied evidence.
- Prefer 1-4 high-signal sentences.

Improved review:
```

## Experiments and ablations run
- **Model sweep**: run the same procedure with different local models (e.g., llama3-style, deepseek-coder, qwen2.5) to see which rewrites score best.
- **Fixed split**: evaluated on the test set of 120 items (40 per language) for comparability.
- **Comparisons**: baseline proposal v1 is measured alongside other strategies (iterative loop, single-edit, rewrite-only, no-evidence, random selection, threshold-gated refinement) on the same data to see relative gains.
