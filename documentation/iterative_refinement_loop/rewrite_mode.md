# Rewrite-From-Scratch Variant

Full rewrite mode that allows larger edits to chase maximum quality.

- **What changes**: prompt instructs the editor to rebuild the review from scratch, emphasizing uncovered claims and key risks/tests instead of minimal edits.
- **Why run it**: tests whether freeing the model from edit constraints yields higher-quality reviews, especially when the seed review is weak.
- **Behavior to note**: loosened change budget but the same relevance and evidence checks apply; selection still favors concise, high-relevance candidates.
