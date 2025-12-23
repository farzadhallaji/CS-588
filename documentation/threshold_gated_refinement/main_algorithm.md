# Threshold-Gated Refinement

Single-pass refinement that only activates when the starting review is too off-target.

- **Gate first**: score the junior review for relevance to the claims; if it meets the threshold, keep it unchanged.
- **Triggering path**: when relevance is below threshold, build a prompt with the current review and claims, then ask a local model for one revised version.
- **Safety checks**: if the model returns nothing, fall back to the seed; improvements are marked when the new relevance exceeds the seed’s.
- **Outputs**: original and refined reviews, their scores, the prompt style used, and whether an improvement occurred.
- **Rationale**: saves budget on already-good reviews while still offering targeted upgrades for weak ones.
