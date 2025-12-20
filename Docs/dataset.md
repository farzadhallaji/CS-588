Use **CRScore/human_study/phase1 + phase2**. Everything else is mostly noise for *this* paper.

### What you should use (for your iterative review-editing paper)

**Main dataset (loop input + pseudo-references):**

- **`CRScore/human_study/phase1/raw_data.json`**
  - Has exactly what your loop needs: diff + full files + human `msg` seed + model reviews + claims/issues/smells + context.
  - This is your **training ground for the loop design** (dev split) and your **test bed** (held-out split).

**Gold human evaluation (eval-only):**

- **`CRScore/human_study/phase2/\*_review_qual_\*.csv`** (or the aggregated `human_eval_rel_scores_thresh_*.json`)
  - This is how you prove you’re not just gaming CRScore.
  - **Never** touch phase2 during tuning/selection/early stopping.

### What you should optionally use (only if you need it)

**Robustness / domain shift (CRScore-only or claim-acc eval):**

- `CRScore/experiments/java_code_smells/*` (and similar smell sets)
  - Great for “noisy claims/smells” + “domain shift” stories.
  - Don’t overcomplicate: use a small subset and report degradation.

**If you insist on training an editor (not required)**

- `msg-{train,valid,test}.jsonl`
  - Only useful if you want to **fine-tune** a small model to write/edit review text.
  - But this changes the paper into “we trained a comment model,” which is a different fight.

### What you should NOT use (for this paper)

- `ref-*.jsonl` → code refinement, not review editing.
- `cls-*.jsonl` → diff quality classification, not your task.
- `dataset.zip` and `CRScore/UniXcoder/downstream-tasks/**` → benchmark zoo; it’ll bloat your scope and reviewers will demand you evaluate on everything.
- `code/data/Diff_Quality_Estimation` → it’s a symlink to nowhere locally, so it’s literally useless unless you fetch the external dataset.
- `all_model_rel_scores_thresh_*.json` / `baseline_metric_scores.json` → fine for analysis plots, not for core experiments.

### Minimal “correct” setup (so reviewers don’t kill you)

1. Build dev/test splits from **phase1 raw_data** (no phase2 used).
2. Run your loop using phase1 claims/issues + evidence lines.
3. Final report: CRScore deltas + **phase2 human quality deltas** on held-out test.

One thing you need to decide (because it changes what datasets matter): **Are you training/fine-tuning the editor, or using frozen small models as editors?** If frozen, you can ignore `msg-*` entirely. 
