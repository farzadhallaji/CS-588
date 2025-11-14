### What CRScore Outputs
CRScore is a reference-free evaluation metric specifically designed to assess the quality of code review comments based on a given code change (diff). It does **not** output a single overall number between 0 and 1. Instead, it provides **three fine-grained scores**, each normalized to the range [0, 1], corresponding to key dimensions of review quality:

- **Comprehensiveness (Comp)**: Measures how thoroughly the review covers all potential issues, claims, or implications in the code change. It's like "recall" in information retrieval—how much of the important stuff is addressed? A score of 1 means the review covers everything it should; 0 means it misses most or all key points.
- **Conciseness (Con)**: Measures how focused and efficient the review is, without unnecessary or redundant content. It's like "precision"—how much of the review is directly on-topic? A score of 1 means every part of the review is relevant; 0 means it's mostly verbose or off-topic.
- **Relevance (Rel)**: This is the **primary overall quality score**, calculated as the harmonic mean of Comp and Con:  
  \[
  \text{Rel} = \frac{2 \cdot \text{Con} \cdot \text{Comp}}{\text{Con} + \text{Comp}}
  \]
  It's analogous to an F1-score, balancing coverage and focus. While not a "single" score in isolation, Rel serves as the holistic indicator of review effectiveness (e.g., a high Rel means the review is both thorough and to-the-point).

In addition to these numerical scores, CRScore can provide **detailed feedback** by highlighting mismatches—e.g., which potential issues (from the code) the review missed or which parts of the review are irrelevant. This feedback is derived from the underlying comparisons but isn't always automatically outputted; it depends on the implementation (e.g., in tools like your proposed pipeline, it could flag gaps for iterative refinement).

For example:
- A verbose review that covers everything might get Comp = 0.9, Con = 0.6, Rel = 0.72.
- A short but incomplete review might get Comp = 0.5, Con = 0.9, Rel = 0.64.

These scores align well with human judgments (Spearman correlation ~0.54), outperforming traditional metrics like BLEU or BERTScore.

### How CRScore Works: Step-by-Step Process
CRScore uses a **neuro-symbolic approach** (combining LLMs for semantic understanding with static analysis tools for rule-based checks) to evaluate reviews without needing "ground truth" human references, which can be noisy or unavailable. It's grounded in the actual code change to ensure objectivity. The full workflow is as follows:

1. **Inputs**:
   - **Code Change (Diff)**: The specific modifications in the code (e.g., added/removed lines in Python, Java, or JavaScript files).
   - **Review Comment**: The natural language text of the review (human- or AI-written), which might include observations, suggestions, or questions.

2. **Pseudo-Reference Generation** (Creating "Expected" Topics):
   - CRScore generates a list of **pseudo-references**—synthetic, verifiable claims about what a good review should address. This avoids reliance on subjective human references.
   - **LLM Component**: A fine-tuned open-source LLM (e.g., Magicoder-S-DS-6.7B, trained on GPT-4 synthetic data) analyzes the diff to produce:
     - **Low-level claims**: Direct facts about changes (e.g., "A new variable 'has_all_data' was added to the table.").
     - **High-level implications**: Broader effects or risks (e.g., "This could lead to null values in existing rows, affecting queries.").
   - **Static Analysis Tools (CATs) Component**: Language-specific tools detect "code smells" and structural issues:
     - Python: PyScent (e.g., unused variables, type conversions).
     - Java: PMD (e.g., cyclomatic complexity, low cohesion).
     - JavaScript: JSHint (e.g., leaking variables, syntax errors).
     - These add objective, tool-verified issues like performance lags or maintainability flaws.
   - **Combination**: LLM claims and tool outputs are merged into a unified list of pseudo-references (e.g., 5–10 items per diff). Human validation shows ~83% accuracy in these pseudo-refs.

3. **Scoring via Semantic Textual Similarity (STS)**:
   - **Embeddings**: Both the pseudo-references and review comment are broken into sentences. Each sentence is embedded into vectors using a sentence transformer model (e.g., mxbai-embed-large-v1), which captures semantic meaning (after removing stopwords).
   - **Similarity Matrix**: Compute pairwise cosine similarity between every pseudo-reference sentence (P) and every review sentence (R). This creates a matrix of scores (0–1 per pair).
   - **Threshold Matching**: A fixed similarity threshold (τ ≈ 0.73, calibrated on datasets) determines if a review sentence "addresses" a pseudo-reference:
     - If similarity > τ, it's a match.
   - **Dimension Calculations**:
     - **Comprehensiveness (Comp)**: Fraction of pseudo-references covered by at least one review sentence (recall-like).
     - **Conciseness (Con)**: Fraction of review sentences that match at least one pseudo-reference (precision-like).
     - **Relevance (Rel)**: Harmonic mean of the above, as the balanced overall score.
   - This process ensures the evaluation is **grounded** (tied to verifiable code facts) and **fine-grained** (sentence-level analysis).

4. **Outputs and Feedback**:
   - The three scores (as described above).
   - **Detailed Feedback**: By inspecting unmatched pseudo-references, you can identify gaps (e.g., "Missed potential null value issue") or irrelevant review parts (e.g., "This sentence on styling is off-topic"). In your project's iterative pipeline, this feedback drives LLM refinements.

#### Example from the Original Paper
- **Code Diff**: Adding a new column `has_all_data` to a database table (allows NULLs).
- **Pseudo-References Generated**:
  - "A new column named ‘has_all_data’ has been added."
  - "This column allows NULL values by default."
  - "Existing rows will not have a value for this column." (LLM claim).
  - "Unused variable detected in related code." (Tool smell).
- **Sample Review Comment**: "This column is not being used anywhere in the codebase. It’s a waste of space."
  - **Scores**: Might get high Con (focused), low Comp (misses NULL implications), moderate Rel.
  - **Feedback**: "Overlooked potential issues with NULL values in queries."

This makes CRScore robust for multilingual code (Python, Java, JS) and better than reference-based metrics, which struggle with diverse phrasings. If you're implementing it (e.g., via the open-source GitHub repo), you can customize τ or tools for your needs. 





____



------------------------------------------------
0.  Inputs you must already have
------------------------------------------------
- A code-review data set that contains  
  – the raw diff (old → new file),  
  – the file content before and after the change,  
  – (optionally) the human-written review that was posted on GitHub.  
  The authors use the公开 “CodeReviewer” split (Li et al. 2022).  
- A pool of review-generation systems you want to measure (their paper tests 9: BM-25, LSTM, CodeReviewer, GPT-3.5, LLaMA-3-8B-Instruct, Magicoder-S-DS-6.7B, …).  
- One reasonably strong LLM that can be *prompted once* to create synthetic training data (they use GPT-4).  
- A smaller open-source code LLM that you will fine-tune (they use Magicoder-S-DS-6.7B).  
- Static-analysis tools that already know how to flag “code smells” for the languages you care about.  
  – Python → PySmell  
  – Java → PMD  
  – JavaScript → JSHint  
- A sentence-transformer model that is good at semantic textual similarity on mixed code+text sentences (they use mixedbread-ai/mxbai-embed-large-v1, ≤1 B params, best on MTEB English STS as of July 2024).

------------------------------------------------
1.  Build a tiny silver training set for claim generation
------------------------------------------------
1.1.  Randomly sample 1 000 diffs from the *validation* split of CodeReviewer.  
1.2.  For every sampled diff, prompt GPT-4 with a carefully engineered zero-shot prompt (their exact prompt is in Appendix D.4).  
      Prompt gist:  
      “You are an expert code reviewer.  
       Given the following unified diff and the full source files, list *all* claims, implications, and code-smell warnings that a good review should mention.  
       Format each item as a single concise sentence.”  
1.3.  Store the resulting list-of-strings as the *silver* pseudo-references for those 1 000 diffs.  
      (You will use these silver labels to fine-tune the small LLM in step 3.)

------------------------------------------------
2.  Fine-tune the small open-source LLM to imitate GPT-4
------------------------------------------------
2.1.  Convert each silver example into a simple instruction–response pair:  
      Instruction = “List claims/implications/smells for the following diff:\n{diff_text}”  
      Response = bullet list of sentences produced by GPT-4.  
2.2.  Train Magicoder-S-DS-6.7B with standard next-token prediction (cross-entropy) for ~3 epochs, early-stop on validation perplexity.  
      → The resulting checkpoint is called “Magicoder-S-DS-6.7B-claims” in their paper.  
      → This model is deterministic and cheap to run at inference time.

------------------------------------------------
3.  Generate pseudo-references for *every* diff in your benchmark
------------------------------------------------
For every diff in the *test* split (≈ 9 869 for CodeReviewer) do:  
3.1.  Run the fine-tuned Magicoder model once → get a list of natural-language claims/implications.  
3.2.  Run the three static-analysis tools on the *after* version of every changed file.  
      – PySmell outputs smell descriptions such as “Long Parameter List”, “Unused Variable”, …  
      – PMD and JSHint output rule-based warnings such as “Unnecessary boxing”, “Leaking global”, …  
3.3.  Merge the two streams:  
      pseudo-references = {LLM_claims} ∪ {CAT_warnings}, after de-duplicating exact string matches.  
3.4.  Store the final list (average size 4.76 sentences) in a jsonl field “p_refs”.

------------------------------------------------
4.  Compute the similarity threshold τ (once, offline)
------------------------------------------------
4.1.  Take the *best* review generation system you have (they use GPT-3.5).  
4.2.  For every sentence in every GPT-3.5 review, embed it with the sentence transformer.  
4.3.  For every sentence, find its most similar pseudo-reference (cosine similarity).  
4.4.  Average all those best-pair similarities → τ_best = 0.7314.  
      (They show the metric is robust to ±0.05 around this value; see Table 18.)

------------------------------------------------
5.  Embed everything
------------------------------------------------
5.1.  Embed every pseudo-reference sentence (stop-words removed, mean-pooled).  
5.2.  Embed every sentence in every candidate review (same preprocessing).  
      Cache these embeddings; you will reuse them for every metric computation.

------------------------------------------------
6.  Metric formulas (the actual CRScore)
------------------------------------------------
Let  
  P = set of pseudo-reference sentences for a diff,  
  R = set of review sentences for that diff,  
  s(·,·) = cosine similarity between embeddings,  
  τ = 0.7314 (or your own τ).

Conciseness (precision-like):  
  Con = Σ_{r∈R} 𝟙[ max_{p∈P} s(p,r) > τ ]  /  |R|

Comprehensiveness (recall-like):  
  Comp = Σ_{p∈P} 𝟙[ max_{r∈R} s(p,r) > τ ]  /  |P|

Overall Relevance (F1-like harmonic mean):  
  Rel = 2 · Con · Comp / (Con + Comp)

All three values lie in [0,1].

------------------------------------------------
7.  Human-validation loops (two annotation stages)
------------------------------------------------
Stage-1  validate the *pseudo-references*  
  – Sample 300 diffs (100 per language).  
  – Two trained annotators (co-authors) code each p-ref:  
    1 = verifiably correct, 0 = verifiably false, –1 = unverifiable.  
  – They may also *add* missing claims.  
  – Results: 82.6 % of LLM claims are correct; κ = 0.80.  
  – Update the universal p-ref list by inserting added claims and deleting false ones.

Stage-2  validate the *review-quality* dimensions  
  – For the same 300 diffs, show annotators: diff + updated p-refs + 10 reviews (9 systems + ground-truth).  
  – Annotators link each review sentence to the p-refs it covers.  
  – Then assign 1–5 Likert scores for conciseness, comprehensiveness, relevance.  
  – Krippendorff’s α > 0.85 for each dimension → reliable gold labels.

------------------------------------------------
8.  Compute correlations and publish
------------------------------------------------
8.1.  Spearman ρ and Kendall τ between Rel and human-relevance score across the 300 items.  
      ρ = 0.543, τ = 0.457 → highest among open-source metrics.  
8.2.  Rank the 9 systems by mean Rel score and by mean human-relevance; compute rank correlation.  
      ρ_rank = 0.95 → essentially the same ordering.  
8.3.  Release:  
      – The 2 900 human relevance scores.  
      – The updated pseudo-references for all test diffs.  
      – Code and scripts to reproduce every number.  
      (All at https://github.com/atharva-naik/CRScore)

------------------------------------------------
9.  How to use CRScore on *your* new model tomorrow
------------------------------------------------
A.  Run your model on every diff → obtain review sentences (R).  
B.  Load the already-computed embeddings for P (or regenerate with the same transformer).  
C.  Embed your review sentences R.  
D.  Compute Con, Comp, Rel with the formulas above.  
E.  Done – no human re-annotation needed, no reference reviews needed, fully reproducible.



____

CRScore is a reference-free metric for evaluating the quality of code review comments based on a given code change (diff). During inference, it computes scores for conciseness, comprehensiveness, and overall relevance by generating pseudo-references (grounded in code claims and smells) and comparing them semantically to the review comments. Here's a step-by-step breakdown of the process:

1. **Generate Pseudo-References (P)**:  
   Use LLMs (like a fine-tuned Magicoder-6.7B) to extract low-level claims (e.g., specific changes like adding a column) and high-level implications (e.g., broader effects on workflows) from the code diff. Complement this with static code analysis tools (CATs) such as PyScent for Python, PMD for Java, or JSHint for JavaScript to detect code smells and issues (e.g., long methods, unused variables, syntax errors). Combine these into a unified set of pseudo-references, which represent the key topics a good review should address.

2. **Prepare Review Sentences (R)**:  
   Split the input code review comment into individual sentences.

3. **Compute Embeddings and Similarities**:  
   Use a sentence transformer model (e.g., mxbai-embed-large-v1) to generate embeddings for each pseudo-reference and each review sentence (after removing stopwords and pooling tokens). Then, calculate pairwise cosine similarities between all pseudo-references (P) and review sentences (R).

4. **Apply Similarity Threshold (τ)**:  
   Filter the similarities using a predefined threshold (typically around 0.73, calibrated from training data) to identify meaningful matches. Only pairs with similarity ≥ τ are considered relevant.

5. **Calculate Conciseness (Con)**:  
   Compute this as the fraction of review sentences that match at least one pseudo-reference above the threshold. It acts like precision, ensuring the review isn't verbose or off-topic.

6. **Calculate Comprehensiveness (Comp)**:  
   Compute this as the fraction of pseudo-references covered by at least one review sentence above the threshold. It acts like recall, ensuring the review addresses most key aspects of the diff.

7. **Calculate Overall Relevance (Rel)**:  
   Derive this as the harmonic mean (F1-like) of Con and Comp to balance the two, providing the primary quality score.

The final output includes the three scores (Con, Comp, Rel), with Rel being the main metric for overall quality assessment.
