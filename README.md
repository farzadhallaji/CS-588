ProposedApproach runner
=======================

What this adds
--------------
- End-to-end iterative review refinement loop (`ProposedApproach/run.py`) aligned with the methodology/experiments docs.
- Supports dev/test splits from `CRScore/human_study/phase1/raw_data.json`, CRScore scoring, evidence-aware guardrails, and ablation modes.
- Editors: offline template editor (default), echo, or a local Ollama model.

Dependencies
------------
- Python 3.9+ with the dependencies from `CRScore/requirements.txt` (`sentence-transformers`, `torch`, etc.).
- CRScore data available locally (already bundled in this repo).
- For Ollama mode: a running local Ollama server and the requested model pulled (e.g., `ollama pull llama3:8b-instruct-q4_0`).

How to run
----------
- Default loop on dev split with template editor:
  ```
  python ProposedApproach/run.py --split dev --limit 5
  ```
- Test split, rewrite ablation, using Ollama:
  ```
  python ProposedApproach/run.py --split test --mode rewrite --editor-type ollama --ollama-model llama3:8b-instruct-q4_0 --limit 10
  ```
- No-evidence ablation:
  ```
  python ProposedApproach/run.py --mode no-evidence --limit 10
  ```
- Random selection ablation (disables CRScore-based selection):
  ```
  python ProposedApproach/run.py --mode no-selection --limit 10
  ```
- Quick smoke test (template editor, 1 dev instance):
  ```
  python ProposedApproach/test.py
  ```
- If you have a local SentenceTransformer path/model, point to it:
  ```
  python ProposedApproach/run.py --model-path /path/to/model --limit 1
  ```
- Adjust evidence threshold if guardrail is too strict:
  ```
  python ProposedApproach/run.py --tau-evidence 0.35 --limit 5
  ```
- Adjust CRScore tau / edit budget if no improvements are selected:
  ```
  python ProposedApproach/run.py --tau 0.7314 --max-change 0.7 --limit 5
  ```
- Local HF LLM editor:
  ```
  HF_MODEL_PATH=/path/to/local/model ./run_local_llm.sh
  ```

Scripts
-------
- `run.sh`: runs main loop + ablations on test split, then evaluates all systems (writes to `results/`).
- `run_local_llm.sh`: runs the loop with a local HuggingFace model as the editor (writes to `results_hf/`).

Outputs
-------
- JSONL at `ProposedApproach/outputs.jsonl` (configurable via `--output`), one record per instance:
  - `best_review` + `best_score` (Con/Comp/Rel)
  - per-iteration history with uncovered refs, offending sentences, guardrail checks, objectives, and timing.

Key flags
---------
- `--mode` toggles ablations: `loop` (default), `k1`, `no-selection` (random pick), `no-evidence`, `rewrite`.
- `--editor-type`: `template` (offline), `ollama`, `echo`.
- `--num-samples`, `--max-iter` tune K/N; `--lang` filters by language; `--limit` bounds run size.
