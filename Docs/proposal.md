- Below is a **full, “submission-ready” research proposal** that (a) matches what you already built (CRScore-guided refinement + evidence guardrails + threshold refine), (b) fixes the obvious evaluation holes (your Phase2 overlap problem), and (c) adds the **one clever ingredient** that lets you run on **other datasets without creating a new dataset**: **automatic claim-bank generation** (diff summary + static analysis “issues” as claims). That’s your bridge to “Q1-ish” territory without begging humans for labels.

  ------

  # Title

  **EviR3: Evidence-Constrained, Critic-Guided Post-Editing for High-Quality Code Review Comments Across Languages**

  ------

  # Abstract

  Large language models can produce code review comments, but they often become vague, miss key change implications, or hallucinate. We propose **EviR3**, a model-agnostic post-editing framework that **improves an existing review comment** given a code patch and lightweight signals about what the change contains. EviR3 combines: (1) a **claim bank** that represents what a good comment should cover (from human annotations when available, otherwise generated automatically from the diff and static analyzers), (2) an automatic **critic** (CRScore-style semantic coverage of claims), (3) an **evidence guardrail** that blocks unsupported additions, and (4) an iterative **search-and-select** refinement loop that proposes multiple candidates and keeps the best under constraints (length, minimal edit, precision preservation). We evaluate on the CRScore human-study benchmark and extend to additional review/comment datasets by generating claims automatically, enabling cross-dataset evaluation without new labeling. We include ablations isolating each component (critic, evidence, selection, rewrite freedom) and a human preference study designed to avoid the “no overlap” pitfall present in existing phase2 ratings.

  ------

  # 1. Motivation

  Code review comments matter because they are the last line of defense before broken code hits production. Humans want comments that are:

  - **Relevant** to the patch (not generic advice),
  - **Comprehensive** (covers main change implications),
  - **Concise** (doesn’t ramble),
  - **Grounded** (no hallucinated claims about code).

  Your current project already shows a key truth: **a small local model can write decent edits if you tell it exactly what’s missing** and enforce constraints with a critic. The problem is: most datasets don’t include the “pseudo-references/claims” that CRScore uses. That’s why a lot of “review generation” papers stall at toy evaluation or subjective examples.

  So the proposal’s core move is:

  > **Make claims cheap to obtain** (from diffs + static analysis), then use your critic-guided refinement to generalize across datasets.

  That’s how you avoid “we created a dataset” as the only path.

  ------

  # 2. Task Definition

  **Input**:

  - A code change ( \Delta ) (unified diff + optionally old file context),
  - An initial review comment ( r_0 ) (human or model-produced),
  - A set of target claims ( C = {c_i} ) describing what the change *does* and what issues matter.

  **Output**:

  - An improved review comment ( r^* ) that:
    1. increases claim coverage and precision (critic score),
    2. avoids unsupported additions (evidence constraint),
    3. stays concise and minimally edited unless rewrite is allowed.

  **Key constraint**:

  - We are doing **post-editing** (review improvement), not free-form generation. This is important because it enables realistic deployment and cleaner evaluation.

  ------

  # 3. Key Idea and Contributions

  ### Contribution 1: Evidence-Constrained Critic-Guided Post-Editing (your loop, formalized)

  We treat review improvement as **discrete search**:

  - Propose candidates using an editor model (local LLM),
  - Score with an automatic critic (CRScore-like),
  - Reject candidates that violate constraints,
  - Select the best and iterate.

  ### Contribution 2: Claim Bank Generation Without Human Labels (the “clever way”)

  When a dataset lacks pseudo-refs/claims, we generate them from:

  1. **Diff summary + implications** (LLM code-change summarization),
  2. **Static analysis findings** (PMD for Java, PyScent for Python, JS linters or lightweight heuristics),
  3. Optional: test changes, API changes, config changes, dependency changes detected from diff patterns.

  These become “claims” the review should address. This converts *any diff dataset* into a claim-based evaluation setting.

  ### Contribution 3: Grounding Guardrail (already in your code)

  A candidate is only accepted if **new sentences are supported** by retrieved evidence snippets from the patch/context (substring or semantic similarity threshold).

  ### Contribution 4: Practical “small compute” recipe

  You explicitly target low-resource hardware (your setup): **6GB VRAM**, **16GB RAM + swap**, Debian, local inference. The method is built to be **model-agnostic** and still effective with quantized 7B–8B editors.

  ------

  # 4. Method: EviR3 Pipeline

  ## 4.1 Claim Bank (C)

  Two modes:

  ### Mode A: Gold/Provided Claims (CRScore phase1 raw_data.json)

  Use existing pseudo-refs/claims from CRScore human study.

  ### Mode B: Generated Claims (for other datasets)

  Generate a claim bank from multiple sources:

  **(i) Diff summarizer (LLM)**
  Prompt a code-change summarizer to output:

  - Main change summary (1–3 bullets),
  - Implications / edge cases / risks,
  - Testing suggestions.

  Then split into atomic claims.

  **(ii) Static analyzers**

  - Java: PMD outputs become claims like “Potential null dereference in X”, “Cyclomatic complexity increased in Y”.
  - Python: PyScent smells (long method, exception smell, cohesion) become claims.
  - JS: lightweight rules or available smell outputs if present.

  **(iii) Heuristic claim templates**
  From diff patterns:

  - “Public API signature changed”
  - “New parameter added / behavior changed”
  - “Config flag now controls behavior”
  - “Error handling changed”
  - “Concurrency / async behavior introduced”

  This makes claim generation robust even if the LLM summarizer is noisy.

  ## 4.2 Evidence Retrieval

  Retrieve short evidence snippets from:

  - patch lines (+/-/context),
  - reconstructed new file window around changed span (you already do this),
  - optionally a few neighboring lines for context.

  Map each claim (c_i) to top-k snippets by lexical overlap (cheap) plus optional embedding similarity.

  ## 4.3 Critic (Automatic Scoring)

  Use CRScore-style semantic alignment:

  - Split review into sentences (S),
  - Compute similarity matrix between claims and sentences,
  - Compute:
    - **Con** (precision-like): fraction of sentences supported by some claim above threshold,
    - **Comp** (recall-like): fraction of claims covered by some sentence above threshold,
    - **Rel** = F1(Con, Comp)

  ### Combined Critic (optional, stronger)

  For generated claims, add **support score**:

  - Penalize sentences not supported by evidence snippets (your guardrail),
  - Penalize excessive copying from claims (your copy penalty),
  - Penalize length.

  Overall objective:
  [
  J(r) = Rel(r) - \lambda_{len}|r| - \lambda_{copy}Copy(r,C)
  ]
  subject to:

  - length ≤ budget,
  - conciseness not dropping beyond (\delta),
  - evidence constraints satisfied,
  - edit distance constraint unless rewrite mode.

  ## 4.4 Refinement Loop (Search)

  Iterate up to T steps:

  1. Compute uncovered claims and offending low-sim sentences,
  2. Build an editor prompt (includes uncovered + offending + evidence),
  3. Sample N candidates,
  4. Filter by constraints,
  5. Select best by objective,
  6. Stop if no improvement for K steps.

  You already implemented this. In the paper, you present it as a constrained search algorithm rather than “prompt engineering”.

  ------

  # 5. Models Suitable for Your Setup (6GB VRAM, 16GB RAM + swap)

  You want “best” for the task under reality, not fantasy.

  ### Editor model (generates candidate rewrites)

  Pick a **code-capable instruction model** in **4-bit quantization**:

  - **Qwen2.5-Coder 7B Instruct (Q4)**: strong for code-related phrasing and technical specificity.
  - **Llama 3/3.1 8B Instruct (Q4)**: generally strong instruction-following and decent editing.
  - **DeepSeek-Coder 6.7B Instruct (Q4)**: strong code prior, often good at review-style comments.

  On 6GB VRAM: run via **Ollama** (GPU if it fits, otherwise CPU). Your method still works if the editor is slower because you can reduce N and T.

  ### Critic model (embeddings)

  Embedding model memory matters. If `mxbai-embed-large-v1` is too heavy on GPU, keep it on CPU. This is fine because scoring is batched and deterministic.

  ------

  # 6. Datasets and Evaluation Plan (No New Dataset Required)

  ## 6.1 Primary Benchmark: CRScore Human Study (Phase1)

  Use the phase1 dataset (your `raw_data.json` already loaded) with 3 languages (py/java/js). Your split scheme (60 dev / 40 test per language) is clean and reproducible.

  This is where you report your main automatic metric improvements because the claim bank exists.

  ## 6.2 Cross-Dataset Generalization Without Labels

  Use existing diff/comment datasets (examples already in the repo tree):

  - `Comment_Generation/msg-*.jsonl` (patch + comment)
  - `Code_Refinement/ref-*.jsonl` (refinement-style data)
  - Any other PR review datasets you can access that include diffs and comments.

  For these datasets:

  - Generate claims automatically (Mode B),
  - Evaluate with the same critic + guardrails,
  - Add a **small human preference study** (Section 7) for credibility.

  This avoids “we created a dataset” while still producing publishable evidence of generalization.

  ------

  # 7. Human Evaluation (Fixing Your Phase2 “No Overlap” Problem)

  Your current Phase2 CSVs don’t overlap systems on the same ids, so paired tests fail. That’s not a “bug”, it’s just bad experimental design for paired stats.

  ### What we do instead (Q1-grade):

  **Human study design:**

  - Sample **K = 30–50 instances per language** from the *same* set of ids (fixed).
  - For each instance, show:
    - patch context (diff + short extracted evidence),
    - baseline comment (r_0),
    - improved comment (r^*) from each method (blinded order).
  - Ask raters for:
    1. Pairwise preference (A vs B),
    2. 1–5 ratings on relevance, completeness, conciseness, groundedness,
    3. “Hallucination” checkbox (claims not supported by diff/evidence).

  **Stats:**

  - Preference win-rate + binomial CI,
  - Wilcoxon signed-rank on per-item ratings (paired, because same ids),
  - Inter-rater agreement (Krippendorff’s alpha or weighted kappa).

  This is cheap enough to run and strong enough to shut reviewers up.

  ------

  # 8. Baselines (What You Must Compare Against)

  You need baselines that match compute and fairness. Here’s a clean baseline set:

  ### B0: Seed comment (do nothing)

  The original human/model review.

  ### B1: Zero-shot rewrite (single pass)

  Prompt the editor to rewrite the review using diff only (no critic loop).

  ### B2: Threshold-gated single-pass refinement (your `threshold_refine.py`)

  Only edit if critic score < threshold.

  ### B3: Few-shot “proposal v1” (your earlier first draft baseline)

  Few-shot examples of low→high quality edits, single pass.

  ### B4: Iterative refinement without selection

  Generate candidates but pick randomly (tests whether scoring/selection matters).

  ### B5: Iterative refinement without evidence guardrail

  Tests hallucination control.

  ### B6: Iterative refinement with rewrite enabled

  Upper bound on quality at the cost of edit faithfulness.

  All of these are already aligned with your repo structure and ablations.

  ------

  # 9. Experiments and Ablations (What You Report)

  ## 9.1 Main results (CRScore phase1 test split)

  Report per language and overall:

  - Rel, Con, Comp (mean),
  - % improved vs baseline,
  - distribution plots (optional but nice).

  ## 9.2 Component ablations

  - No evidence guardrail
  - No selection (random)
  - k=1 (single iteration)
  - rewrite vs minimal-edit constraint
  - different N candidates (1, 2, 4)
  - different thresholds τ and refine-threshold

  ## 9.3 Claim-bank robustness (the part reviewers like)

  On datasets with generated claims:

  - Compare diff-summary claims only vs static-analysis claims only vs combined claims,
  - Inject noise into claims (drop 20%, add irrelevant 20%) and show your loop is stable.

  ## 9.4 Hallucination / groundedness analysis

  Measure:

  - % new sentences rejected by guardrail,
  - human-labeled hallucination rate (from study),
  - correlation between critic score and human relevance.

  ------

  # 10. Prompts for LLM Ablations (Ready-to-Use)

  Below are **drop-in prompts** you can use for `threshold_refine.py` and also as editor instructions for your loop. They’re designed so each variant tests a hypothesis (not vibes).

  ## Prompt Variant: `default`

  **Goal:** general improvement, grounded.

  ```
  You are a senior code reviewer. Improve the review comment.
  
  Requirements:
  - Be specific to the code change and the provided claims/evidence.
  - Do NOT invent facts not supported by the diff/evidence.
  - Prefer concrete correctness, edge cases, and tests over style nits.
  - Keep it concise (2–5 sentences max).
  
  Return ONLY the improved review text.
  ```

  ## Prompt Variant: `concise`

  **Goal:** shorten + remove fluff.

  ```
  Tighten the review to only the highest-signal points tied to the change.
  - Remove generic advice and repetition.
  - Keep only concrete risks, correctness issues, and essential tests.
  - Do not add anything not supported by the diff/evidence.
  
  Return ONLY the revised review.
  ```

  ## Prompt Variant: `evidence_strict`

  **Goal:** maximum grounding (anti-hallucination).

  ```
  Revise the review using ONLY information supported by the provided evidence snippets or directly implied by the diff.
  - If you cannot support a statement, omit it.
  - Prefer referencing specific changed behavior, parameters, edge cases, or tests.
  
  Return ONLY the revised review text.
  ```

  ## Prompt Variant: `test_heavy`

  **Goal:** push actionable test suggestions (your “test-heavy”).

  ```
  Improve the review with a focus on testing and edge cases.
  - Identify what should be tested given the change (positive + negative cases).
  - Avoid generic "add tests" phrasing; propose concrete cases.
  - Do not invent behavior not supported by the diff/evidence.
  
  Return ONLY the improved review.
  ```

  ## Prompt Variant: `bug_hunt`

  **Goal:** correctness and safety.

  ```
  Improve the review focusing on correctness, safety, and failure modes.
  - Call out likely breakpoints: null handling, boundary conditions, concurrency, backward compatibility, security-impacting changes.
  - If nothing is risky, say so briefly and suggest the single most relevant test.
  
  Return ONLY the revised review.
  ```

  ## Prompt Variant: `minimal_edit`

  **Goal:** stress-test “edit not rewrite”.

  ```
  Make MINIMAL edits to the existing review:
  - Keep original intent and wording as much as possible.
  - Only delete irrelevant sentences and add the smallest missing specifics needed to cover the key claims.
  - Do not expand length significantly.
  
  Return ONLY the edited review.
  ```

  ------

  # 11. Expected Outcomes (Grounded in Your Current State)

  You already observed:

  - Threshold-gated refinement with a decent coder model can push Rel up substantially on CRScore phase1.
  - Phase2 evaluation failed **not because your method is bad**, but because the provided CSVs don’t support paired comparisons.

  So the plan is:

  1. Use CRScore phase1 for clean automatic evaluation (you already do this).
  2. Add claim generation to run on other datasets without labels.
  3. Run a small, properly-overlapped human study for “real” validation.

  That’s a coherent paper story.

  ------

  # 12. Threats to Validity (You Must Say This)

  - **Metric overfitting:** optimizing CRScore might produce comments that “game” claims. Mitigation: evidence guardrail + human preference study + hallucination labels.
  - **Claim quality dependence:** generated claims may be noisy. Mitigation: multi-source claims + robustness experiments with injected noise.
  - **Generalization:** datasets vary in diff format and context. Mitigation: evaluate across at least 2 additional datasets besides CRScore phase1.

  ------

  # 13. Ethics and Safety

  - Do not include proprietary code in released artifacts.
  - If using public PRs, respect licensing and remove sensitive identifiers.
  - Evidence guardrails reduce hallucinations and unsafe suggestions.

  ------

  # 14. Reproducibility Plan

  - Fixed splits (seeded), publish split ids.
  - Release prompts, hyperparameters, and exact selection objective.
  - Provide a CPU-only configuration that reproduces main results (important for reviewers).

  ------

  # 15. Paper Structure (What You Actually Submit)

  1. Introduction + why post-editing beats generation
  2. Problem definition + claim bank concept
  3. EviR3 method (critic + evidence + search)
  4. Claim generation for cross-dataset transfer
  5. Experiments on CRScore phase1 + ablations
  6. Cross-dataset results with generated claims
  7. Human study (paired preference + hallucination)
  8. Limitations + ethics + conclusion

  ------

  If you follow this proposal, you’ll stop looking like “we wrapped a metric around an LLM” and start looking like:
  **“we built a general post-editing framework with grounding + scalable evaluation across datasets.”** That’s the difference between a course project and something reviewers might not immediately throw into the trash.
