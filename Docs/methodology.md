## Methodology

### 1. Task definition

We study **iterative refinement of code review comments**. Given a code change (diff + old/new files) and an initial review message $r^{(0)}$ (typically a **human-written** review), the system produces an improved review $r^*$ by repeatedly editing the text under **evidence constraints**. Improvement is measured primarily by **CRScore** (Conciseness/Comprehensiveness/Relevance) and validated by **human judgments** on held-out labels.

------

### 2. Data, splits, and leakage control

**Core dataset.** We use the CRScore human study bundle derived from CodeReviewer: 300 code changes across Python/Java/JavaScript, each containing (i) the diff and full old/new files, (ii) a gold human review message `msg`, (iii) model-generated reviews, and (iv) pseudo-references in the form of **claims/issues/smells** (when available). Human annotations include claim accuracy (phase1) and review quality (phase2), where phase2 is evaluation-only.

**Splits.**

- **Dev (tuning only):** 60 diffs per programming language, sampled from phase1 raw data. Used to tune prompts, $\tau$, loop depth $K$, sampling $N$, and objective weights. **No phase2 labels** are used here.
- **Test (final reporting):** 40 diffs per language. Phase2 labels used **only** for final evaluation.
- **Robustness (transfer):** leave-one-language-out: tune on two languages, test on the third.

**Leakage rule.** Phase2 labels never appear in prompts, selection, early stopping, or parameter tuning. Only CRScore signals computed from pseudo-references and code evidence drive the loop.

------

### 3. Pseudo-references and evidence anchoring

CRScore evaluates a review against a set of **pseudo-references** $P$ (claims/smells/issues). CRScore itself operationalizes relevance via sentence-level semantic similarity against these pseudo-references and computes Conciseness/Comprehensiveness/Relevance (harmonic mean) using a similarity threshold $\tau$.

We support three pseudo-reference sources:

1. **Human-provided** pseudo-references (when present in raw data).
2. **Auto-extracted** pseudo-references using claim/smell extractors (cached). Smells can be produced by language-specific static analyzers where applicable.
3. **None** (ablation): CRScore computed only on whatever minimal signals exist (expected to underperform, included for attribution).

**Evidence retrieval (code grounding).**
For each pseudo-reference $p \in P$, we build a small evidence set $E(p)$ consisting of code lines from the diff/old/new files:

- If the dataset provides supporting lines, we use them directly.
- Otherwise, we retrieve top-$m$ lines by a hybrid signal:
  - lexical overlap between claim text and diff hunks/identifiers,
  - proximity to changed lines,
  - optional embedding similarity between claim and code-line string (cheap sentence embedding).

This evidence is used **only to constrain edits** (Section 5), not as a training signal.

------

### 4. CRScore computation

We compute CRScore on each candidate review $r$ against pseudo-references $P$:

- Split $r$ into sentences $R$.
- Compute semantic similarity $s(p, r_i)$ for all $p \in P$, $r_i \in R$.
- Compute:
  - **Conciseness (Con):** fraction of review sentences that match any pseudo-reference above $\tau$ (precision-like).
  - **Comprehensiveness (Comp):** fraction of pseudo-references matched by any review sentence above $\tau$ (recall-like).
  - **Overall Relevance (Rel):** harmonic mean of Con and Comp.

**Threshold $\tau$.** We select $\tau$ on dev only, then freeze it. CRScore reports a typical $\tau_{best}$ computed from pseudo-reference/review similarity distributions and notes robustness to slight variations.
We also report sensitivity by re-evaluating test outputs under multiple $\tau$ values as a robustness check.

------

### 5. Iterative CRScore-guided editing loop

#### 5.1 Overview

We refine an initial review $r^{(0)}$ using a small LLM editor $M$. At each iteration $t$, we:

1. score $r^{(t)}$ with CRScore to identify uncovered/high-gap pseudo-references,
2. construct an **edit prompt** with uncovered items + evidence snippets,
3. generate $N$ minimally edited candidates,
4. filter candidates with **guardrails** (evidence and precision constraints),
5. select the best candidate by a shaped objective,
6. early-stop if improvement is negligible.

#### 5.2 Edit prompt construction

We extract:

- uncovered pseudo-references: $U^{(t)} = \{p \in P \mid \max_{r_i \in R^{(t)}} s(p, r_i) \le \tau\}$
- “offending” sentences: sentences in $r^{(t)}$ with low maximum similarity to any $p$ (verbosity/noise).
- evidence sets: $\{E(p)\}_{p \in U^{(t)}}$

The editor prompt contains:

- the current review $r^{(t)}$,
- a short list of uncovered pseudo-references (bulleted, optionally ranked by importance),
- for each uncovered item, a small evidence block (diff lines),
- explicit constraints:
  - produce **minimal edits** (insert/delete/replace specific sentences),
  - keep within a length budget,
  - do not introduce new claims not supported by evidence,
  - output the revised review only (or revised review + edit list for logging).

#### 5.3 Minimal-edit enforcement

We enforce “edit, don’t rewrite” via:

- prompting (edit operations),
- and a hard filter: reject candidates with edit distance or sentence-level change ratio exceeding a dev-tuned limit (e.g., $>$40% sentence replacements), except in the “rewrite” ablation.

#### 5.4 Candidate filtering guardrails (anti-metric-gaming)

A candidate $r'$ is rejected if any of the following fails:

1. **Evidence support constraint:** any *new* sentence must have similarity above a threshold $\tau_e$ to at least one evidence snippet $E(p)$ (embedding similarity or lexical match). This blocks evidence-free additions.
2. **Precision floor:** $\text{Con}(r')$ must not drop below $\text{Con}(r^{(t)}) - \delta$ (dev-tuned), preventing “Comp inflation by babbling.”
3. **Copy/parrot penalty:** penalize high overlap between $r'$ and the pseudo-reference text (n-gram overlap or embedding near-duplicate), discouraging trivial restatement.
4. **Length budget:** cap total tokens/characters.

#### 5.5 Selection objective

Among valid candidates, we select:
$$
r^{(t+1)} = \arg\max_{r' \in \mathcal{C}} \big(\text{Rel}(r') - \lambda_{len}\cdot \text{Len}(r') - \lambda_{copy}\cdot \text{Copy}(r')\big)
$$
with optional constraints like $\text{Con} \ge c_{\text{min}}$, $\text{Comp} \ge p_{\text{min}}$ for “min-threshold” variants.

#### 5.6 Iteration and early stopping

Run up to $K$ iterations (e.g., $K \in \{1,2,3\}$). Early stop if:

- $\Delta \text{Rel} < \epsilon$ for two consecutive iterations, or
- no candidate passes guardrails.

We output the best-scoring review across all iterations, not necessarily the last.

------

### 6. Models and generation settings

**Editors (small models).** We evaluate a set of small/code-tuned LLMs as editors (e.g., 3B–13B range), run locally with identical decoding APIs.

**Decoding.**

- temperature and top-p fixed per experiment, tuned on dev.
- For sampling experiments, generate $N \in \{1,2,4,8\}$ candidates per iteration.

We log token counts and wall-clock latency per generation call.

------

### 7. Baselines (fair, information-matched)

To avoid giving the loop an unfair advantage, any baseline that competes with the loop receives the **same pseudo-references and evidence snippets** when applicable.

1. **Original human review** $r^{(0)}$ (no edits).
2. **Single-pass evidence-aware edit:** one call to $M$ with the same edit prompt, $K=1$.
3. **Single-pass evidence-aware rewrite:** one call to $M$ with rewrite permission (edit constraints removed).
4. **Multi-sample single-pass:** generate $N$ candidates in one shot; select by non-CRScore heuristic (e.g., shortest valid) to isolate the effect of CRScore-guided selection.
5. **Loop without CRScore selection:** iterate but choose randomly among valid candidates, isolating the selection signal.

(Optional upper bounds: larger models run once; and human reference review.)

------

### 8. Evaluation protocol

#### 8.1 Automatic metrics

- **Primary:** CRScore Con/Comp/Rel (P/R/F-like behavior).
- **Secondary:** length, edit distance vs. $r^{(0)}$, and coverage statistics (# pseudo-references newly covered).

We additionally report CRScore sensitivity across $\tau$ values (robustness is expected but must be verified in our setting).

#### 8.2 Human-aligned evaluation (held-out phase2)

On the test split, we measure:

- correlation between CRScore improvements and human relevance/quality scores,
- paired comparisons (before vs after refinement),
- system ranking agreement (optional, if multiple systems are compared).

CRScore reports strong ranking correlations with human judgments using Kendall/Spearman in its validation; we follow the same style of analysis.

#### 8.3 Significance testing

For per-diff paired comparisons (baseline vs refined):

- Wilcoxon signed-rank (nonparametric) or paired bootstrap over diffs,
- report effect sizes and 95% confidence intervals.

Multiple comparisons are corrected (Holm–Bonferroni) for ablations.

------

### 9. Robustness experiments

We test robustness along the dimensions reviewers actually care about:

1. **Pseudo-reference noise:** randomly drop 10/20/30% of pseudo-references and inject false positives; measure degradation.
2. **Extractor mismatch:** compare human-provided vs auto-extracted vs none.
3. **Language transfer:** leave-one-language-out tuning.
4. **Domain shift (optional):** evaluate on an unlabeled out-of-domain set with CRScore-only reporting, plus qualitative error analysis.

We also explicitly track known CRScore failure modes (e.g., cases with few claims or heavy inline code) during qualitative analysis.

------

### 10. Cost and latency reporting

For every configuration we report:

- total tokens generated (prompt + completion),
- number of CRScore evaluations (includes selection overhead),
- wall-clock time per diff,
- **Pareto curves** of human-aligned quality gain vs. cost.

We separate:

- editor cost,
- scoring cost (embedding calls + extraction),
  so the “small model is practical” claim is measurable.

------

### 11. Reproducibility

- deterministic seeds for decoding and sampling,
- frozen dev-tuned hyperparameters for test,
- JSONL traces per diff: prompts, candidates, CRScore per iteration, reasons for rejection, selected output, tokens, latency,
- exact dataset IDs per split, plus SHA/versioning for code and model weights.

------
