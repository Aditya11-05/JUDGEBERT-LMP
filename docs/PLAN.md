# Plan and status

## Environment (sandbox used so far)
Python 3.12, 1 CPU, 3 GB RAM, no GPU. PyPI and GitHub reachable; huggingface.co and the PyTorch wheel index are NOT. So XLM-R
fine-tuning and the sentence-embedding baseline cannot be run there; they are written to run on the team's machine / Colab
and are marked pending.

## Conflicts and decisions to confirm
1. Deck writes `[CLS] x [SEP] y [SEP]`; XLM-R uses `<s> x </s></s> y </s>`. Use the tokenizer's pair encoding.
2. Deck says "pooled output"; XLM-R's pooler is not pretrained. Use the `<s>` hidden state + dropout + linear head.
3. Brief/deck treat explanations as optional; the master prompt makes them a contribution. Kept as a separate rule module.
4. Python 3.11 requested; the sandbox has 3.12 (`requires-python >=3.11`).
5. Indian language: Hindi is a candidate; confirm source availability before fixing it.

## Critical path
300+ human-rated pairs. Code cannot replace this: the team must source licensed text, write simplifications, and rate
independently (2+ raters). Until then metrics stay placeholders.

## Status log
* Data + TF-IDF baseline: code and toy-data tests done; real data pending.
* Explanations: English rule module done (docs/EXPLANATIONS.md). Hindi pending. No accuracy measured on human labels.
* FastAPI backend: done and verified offline. Transformer inference and the embedding baseline remain unavailable by design.
* Streamlit frontend: done; verified with AppTest against the real backend and by starting the real server. Not checked in a browser.
* Incident: the sandbox was reset once between milestones and the repository was rebuilt from recorded steps; it was verified
  against the previously reported baseline (297 passed, 1 skipped) before new work. Keep a copy outside the sandbox.
* Next: model/embedding work on a machine with Hugging Face access, and the human-rated dataset.

## Model pipeline implementation (2026-11-16)
* Added `app/models/xlmr.py`: XLM-R pair regressor using tokenizer pair encoding and a bounded 1-10 regression head.
* Added `scripts/train_lmp.py`: trains only on `adjudicated_rating` for human-written/expert-reviewed pairs, uses source-document grouped splits, selects by validation RMSE, and reports held-out test metrics.
* Replaced the inference stub with an artifact loader and batched predictor. The manifest must include model weights and a saved tokenizer/config directory.
* Training has NOT been run: no adjudicated human-rating dataset is currently included. Do not report model performance until a real, held-out, document-grouped evaluation is completed.
