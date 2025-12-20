Here’s the **minimum set** that can still pass top-tier reviewer cruelty. Anything less and they’ll (correctly) say you’re just optimizing your own metric.

## Minimal experiments (3 tables + 1 figure)

### **Table 1: Main results (the paper’s spine)**

**Goal:** prove the loop improves *human-written* reviews and it’s not just CRScore masturbation.

Systems (minimum fair set):

1. **Human seed** (original `msg`)
2. **Single-pass evidence-aware edit** (same claims/smells + evidence, one shot)
3. **Single-pass evidence-aware rewrite** (same info, but free rewrite)
4. **Iterative loop** (your method, fixed (K), (N))

Report on **test only**:

- CRScore: Con / Comp / Rel
- Human phase2: Rel(F) (and P/R if you have them)
- Paired significance: Wilcoxon or paired bootstrap over diffs

**If you don’t include #2 and #3, reviewers will say your loop wins only because you gave it more structure or more chances.**

------

### **Table 2: Minimal ablations (prove causality, not vibes)**

**Goal:** isolate what actually matters. Keep it brutal and small.

Run these variants of *your loop* (same (K,N)):
A) **No iteration**: (K=1) (kills “iteration matters” if it doesn’t drop)
B) **No CRScore selection**: sample (N), pick random/shortest valid (tests “metric-guided selection matters”)
C) **No evidence constraint**: remove evidence anchoring/guardrail (tests “grounding prevents hallucination + matters for quality”)
D) **Rewrite instead of edit**: remove minimal-edit constraint (tests “editing vs rewriting” claim)

That’s it. Four ablations. Anything more belongs in an appendix.

------

### **Table 3: Robustness (one clean stress test)**

**Goal:** show it doesn’t collapse when pseudo-references are imperfect (they are).

Pick **one** noise model (minimum):

- **Dropout 20% and 30%** of claims/smells (randomly), evaluate degradation.

Optional but high-value (still minimal):

- Inject **10% false positives** (nonsense claims). If your method starts parroting, you’ll see it.

Report deltas vs non-noise. Include one paragraph of failure modes.

------

### **Figure 1: Pareto frontier (quality vs cost)**

**Goal:** justify the “small models are practical” angle with one figure.

For your **best small editor model only**:

- Sweep ((K,N)): e.g. (K \in {1,2,3}), (N \in {1,2,4})
- x-axis: cost (tokens + wall time)
- y-axis: Human Rel(F) (preferable) or CRScore Rel if you must

This figure sells the paper to anyone who cares about systems.

------

## Minimal ablations (recap)

If you only do **four**, do these:

1. **K=1** (no iteration)
2. **No CRScore selection** (random/shortest pick)
3. **No evidence constraint** (no grounding guardrail)
4. **Rewrite mode** (no minimal-edit constraint)

These correspond to the four claims reviewers will attack:

- “iteration matters”
- “CRScore selection matters”
- “evidence prevents hallucination and improves quality”
- “editing is better than rewriting for human reviews”

------

## What you can drop without dying

- BM25/LSTM baselines (nobody cares if your real baselines are strong)
- 10 prompt formats (pick one good one, done)
- claims-only vs smells-only (nice but not required for safe accept)
- cross-model sweep (do in appendix if you have cycles)

------

## Non-negotiable fairness rule

Every baseline that competes with your loop must get the **same inputs** (claims/smells + evidence). Otherwise the paper is dead on arrival.

That’s the minimal package that still looks like real research instead of “we tuned a loop to win our own score.” 
