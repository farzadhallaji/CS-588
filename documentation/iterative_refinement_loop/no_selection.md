# No-Selection Variant

Ablation that removes the objective-driven selector to test its importance.

- **What changes**: after filtering invalid candidates, one valid revision is picked without using the relevance/penalty objective (e.g., random or shortest admissible).
- **Why run it**: measures how much the scoring-based selector contributes beyond the candidate generator itself.
- **Behavior to note**: still enforces evidence support and conciseness guards; quality differences isolate the selector’s impact.
