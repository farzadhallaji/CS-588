# Proposal v1 Few-Shot Baseline

Single-pass rewrite guided by example pairs and automatic feedback on the seed review.

- **Inputs**: junior review, code diff, pseudo-reference claims, and a small set of paired examples showing weak vs improved reviews.
- **Feedback-driven prompt**: the seed review is first scored on relevance, conciseness, and comprehensiveness; those scores go into the prompt with the examples to steer edits (add missing points, cut fluff, align to the change).
- **Rewrite once**: a local instruction-tuned model produces one improved review at low temperature to stay focused; if it returns nothing, keep the seed review.
- **Rescore and log**: the improved review is rescored on the same dimensions to quantify gains; outputs include seed/final text, scores, and a method tag.
- **Intended use**: establishes a transparent baseline that mixes human-crafted examples with automatic critique, useful for comparing against iterative loops or rerank pipelines.

## Prompt text
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
