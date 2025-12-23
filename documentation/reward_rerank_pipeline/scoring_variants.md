# Scoring and Reward Variants

Different reward settings stress which penalties matter and how strict the evidence threshold should be.

- **Penalty toggles**: compare full reward (relevance minus unsupported/length/copy penalties) against ablations that disable one or more penalties to see their contribution.
- **Soft vs hard matching**: soft mode uses continuous similarity; hard mode treats matches as pass/fail at the similarity threshold, producing sharper filtering.
- **Threshold sweeps**: adjust the similarity threshold to be more forgiving or strict, revealing how sensitive selection is to evidence strictness.
- **Selection temperature**: vary the randomness used when sampling from reward scores to test stability under near-ties.
- **Weight tuning**: reward weights for unsupported claims, length, and copying can be rebalanced; runs log the configuration used so improvements can be traced back to weight choices.
