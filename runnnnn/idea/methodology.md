## Methodology

In this section, we delineate the methodological framework underpinning CRScore-Loss, our differentiable adaptation of the CRScore metric for fine-tuning large language models (LLMs) in code review generation. We commence with an overview of the core components, followed by a rigorous exposition of the metric's reformulation, the integration of code smell signals, and the fine-tuning pipeline. Our approach is designed to optimize LLM outputs for relevance, comprehensiveness, and factual grounding, leveraging gradient-based learning while preserving the domain-specific insights of the original metric. All implementations are built upon PyTorch for differentiability and Hugging Face Transformers for model handling, ensuring reproducibility and scalability.

### Overview of CRScore-Loss Framework

CRScore, as introduced by Naik et al. (2024), evaluates code review comments reference-free by deriving pseudo-references from code diffs—comprising *claims* (factual statements about changes, generated via an LLM like Magicoder) and *smells* (design flaws detected through static analysis tools)—and computing precision (conciseness), recall (comprehensiveness), and F1 (relevance) via thresholded cosine similarities using sentence embeddings. While effective for evaluation, its discrete nature (e.g., binary matching via max pooling and hard thresholding) precludes direct use as a loss function due to non-differentiability.

To address this, CRScore-Loss reformulates these elements into a continuous objective. The framework consists of three phases: (1) pseudo-reference generation, (2) soft similarity computation, and (3) loss aggregation with auxiliary terms. During fine-tuning, an LLM generates candidate reviews for a given diff, which are scored against pseudo-references to guide optimization. We employ a hybrid supervised-reinforcement setup, combining cross-entropy loss for initial alignment with policy gradients for reward maximization, ensuring stable training.

Formally, for a code diff \( d \), pseudo-references \( P = C \cup S \) (where \( C \) are claims and \( S \) are smells), and generated review sentences \( R = \{r_1, \dots, r_m\} \), CRScore-Loss \( \mathcal{L} \) is defined as:
\[
\mathcal{L} = -\left( \alpha \cdot \mathcal{P}(R, P) + \beta \cdot \mathcal{R}(R, P) + \gamma \cdot \mathcal{S}(S) \right),
\]
where \( \mathcal{P} \) and \( \mathcal{R} \) are soft precision and recall, \( \mathcal{S} \) is a smell penalty, and \( \alpha, \beta, \gamma \) are hyperparameters (set to 0.4, 0.4, 0.2 via grid search).

### Pseudo-Reference Generation

Pseudo-references form the grounding for our loss, mirroring CRScore but optimized for training efficiency.

#### Claim Extraction
Claims are generated using a pre-trained LLM (Magicoder-S-DS-6.7B) prompted to summarize diffs into factual bullets. The prompt is:
```
Summarize the code change in the following diff, focusing on low-level modifications (e.g., variable changes, control flow alterations) and high-level implications (e.g., performance impacts, security risks):
{diff}
Output as a bullet list of claims.
```
For diffs \( d \), we generate \( C = \{c_1, \dots, c_k\} \) with \( k \approx 5-15 \) depending on diff complexity. To handle empty diffs (as in some datasets), we fall back to code snippets if available. Generation uses beam search (beam=4) for diversity, with outputs parsed via regex to extract bullets starting with "- " or numerics.

#### Smell Detection
Smells \( S \) are extracted using language-specific analyzers: PMD (v7.6.0) for Java with rulesets for design, best practices, and error-prone categories; Pyscent for Python, detecting issues like long methods and cyclomatic complexity; and JSNose (customized from the repo) for JavaScript. For a diff, we compute differential smells: \( S = S_{\text{new}} \setminus S_{\text{old}} \), where old/new code is parsed from diff hunks. If no diff, we analyze the full snippet as "new." Outputs are textual descriptions (e.g., "Long Method: Method exceeds 50 lines"), ensuring compatibility with embedding-based scoring.

Pseudo-references are embedded using mxbai-embed-large-v1, a high-performance sentence transformer yielding 1024-dimensional vectors, pre-computed for efficiency during training.

### Differentiable Reformulation of CRScore

To enable gradients, we soften CRScore's components while preserving its semantic intent.

#### Soft Similarity Matrix
Given embeddings \( \mathbf{P} \in \mathbb{R}^{n \times d} \) for \( P \) (n pseudo-refs) and \( \mathbf{R} \in \mathbb{R}^{m \times d} \) for \( R \) (m review sentences), the similarity matrix \( \mathbf{S} = \cos(\mathbf{R}, \mathbf{P}) \) is computed as in CRScore. Instead of hard max-thresholding, we apply a sigmoid activation for soft matching:
\[
\sigma(x) = \frac{1}{1 + e^{-\kappa (x - \tau)}},
\]
where \( \tau = 0.7314 \) (from CRScore) and \( \kappa = 10 \) (sharpness hyperparameter, tuned for gradient stability).

#### Soft Precision and Recall
Precision (conciseness: fraction of review sentences matching at least one pseudo-ref) becomes:
\[
\mathcal{P}(R, P) = \frac{1}{m} \sum_{i=1}^m \max_j \sigma(S_{i,j}),
\]
approximating the hard max with a soft-max variant for differentiability:
\[
\max_j \sigma(S_{i,j}) \approx \frac{\sum_j \sigma(S_{i,j}) \exp(\lambda \sigma(S_{i,j}))}{\sum_j \exp(\lambda \sigma(S_{i,j}))},
\]
with \( \lambda = 5 \) (temperature). Recall (comprehensiveness) is symmetrically:
\[
\mathcal{R}(R, P) = \frac{1}{n} \sum_{j=1}^n \max_i \sigma(S_{i,j}),
\]
using the same soft-max.

#### Auxiliary Smell Penalty
To emphasize smell coverage, we add \( \mathcal{S}(S) = -\frac{1}{|S|} \sum_{s \in S} \max_{r \in R} \cos(\mathbf{e}_s, \mathbf{e}_r) \), a negative mean similarity encouraging the model to address smells explicitly. This term is scaled by \( \gamma \) to prevent dominance over claim-based signals.

### Fine-Tuning Pipeline

We fine-tune a base LLM (e.g., CodeLlama-7B) on code review datasets using a two-stage process.

#### Stage 1: Supervised Alignment
Initialize with supervised fine-tuning (SFT) on pairs (diff, gold review) using cross-entropy loss \( \mathcal{L}_{\text{SFT}} = -\sum \log p(r | d) \), where \( r \) is the tokenized review. This provides a stable starting point, trained for 2-3 epochs with LoRA (r=16, α=32) on 8xA100 GPUs, batch size 16, learning rate 1e-4 via AdamW.

#### Stage 2: Reinforcement with CRScore-Loss
Transition to RLHF using PPO, where CRScore-Loss serves as the reward \( r_t = - \mathcal{L} \). For each episode, sample reviews from the policy \( \pi_\theta \), compute rewards via CRScore-Loss, and update with clipped surrogate objective. We incorporate a KL-divergence penalty (β=0.02) to prevent mode collapse. Training spans 1-2 epochs, with value head fine-tuned alongside the policy.

#### Hyperparameters and Ablations
Key hyperparameters (α, β, γ, κ, λ) are tuned via grid search on a validation split, evaluating against human annotations. Ablations assess the impact of soft components (e.g., hard vs. soft max) and smell integration, ensuring robustness.

This methodology ensures CRScore-Loss is not only theoretically sound but empirically verifiable, facilitating seamless integration into LLM pipelines for enhanced code review automation. 
