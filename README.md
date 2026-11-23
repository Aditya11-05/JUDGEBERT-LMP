# JUDGEBERT-LMP (research prototype)

JUDGEBERT-inspired **legal meaning preservation (LMP)** scoring for English and one Indian language (Hindi is a candidate,
not confirmed). CSET 346, Bennett University. This is **not** the official JUDGEBERT model and gives **no legal advice**.
Original work: Beauchemin et al. (2025), https://aclanthology.org/2025.emnlp-main.5.pdf ; code: https://github.com/GRAAL-Research/JUDGEBERT.
Nothing reported by that paper is a result of this repository.

## Status (honest)
| Component | State |
|---|---|
| Dataset schema, validation, duplicate detection, group-aware split + leakage check | implemented, tested |
| TF-IDF baseline + calibration (train-only fit), metrics, artifact save/load | implemented, tested on toy data only |
| Bounded score `1+9*sigmoid(z)`, configurable categories | implemented, tested |
| Rule-based legal-error explanations (English) | implemented, tested; accuracy on human labels not measured |
| FastAPI backend (`/health`, `/predict`, `/batch-predict`, `/model-info`) | implemented; verified with no model, a mock model and toy baselines |
| **Streamlit frontend** (analyzer, batch CSV, status, research info) | **implemented; verified with Streamlit's AppTest against the real backend over HTTP, and the `streamlit run` server starts. Not checked in a real browser.** |
| Hindi (explanations, API, UI) | **not implemented**; rejected by the API, disabled in the UI |
| Trained XLM-R regressor and transformer inference | **not started** (needs Hugging Face + PyTorch access); everything reports it unavailable |
| Multilingual-embedding baseline | **not started**; reported unavailable |
| Human-rated dataset (>=300 pairs, >=2 annotators) | **does not exist yet** |
| Experiments 1-5, IEEE report | pending |

**Right now no LMP score can be produced anywhere**, because no model is trained. The API and UI show an explicit
"unavailable" state plus the real rule-based findings. No evaluation numbers are reported because none exist.

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate      # macOS / Linux, Python >= 3.11
pip install -r requirements.txt                          # backend + tests
pip install -r requirements-ui.txt                       # Streamlit frontend (kept separate on purpose)
```
(`pip install -e ".[dev]"` and `".[ui]"` are equivalent extras.) `torch`/`transformers` (extra `ml`) are not needed yet.

## Run the application (two terminals, from the repository root)
```bash
# 1. backend (http://localhost:8000, docs at /docs)
uvicorn app.main:app --port 8000

# 2. frontend (http://localhost:8501)
streamlit run frontend/streamlit_app.py
```
Run Streamlit from the repository root so `.streamlit/config.toml` (light theme, 5 MB upload cap) is picked up.

### Environment variables (all optional; see `.env.example`)
| Variable | Used by | Default |
|---|---|---|
| `JUDGEBERT_API_URL` | frontend | `http://localhost:8000` |
| `JUDGEBERT_UI_MAX_ROWS` / `JUDGEBERT_UI_MAX_CHARS` | frontend CSV limits | `2000` / `5000` |
| `JUDGEBERT_MODEL_DIR`, `JUDGEBERT_BASELINE_DIR`, `JUDGEBERT_RESULTS_DIR`, `JUDGEBERT_CONFIG` | backend | `models/lmp`, `models/baselines`, `reports/results`, `configs/default.yaml` |
| `JUDGEBERT_MAX_CHARS`, `JUDGEBERT_MAX_BATCH`, `JUDGEBERT_MAX_BODY_BYTES`, `JUDGEBERT_PRESERVED_MIN`, `JUDGEBERT_PARTIAL_MIN` | backend | `5000`, `100`, `2000000`, YAML |

### Backend health is not model availability
`GET /health` returning `status: "ok"` only means the web service is up. The model is usable only when
`ready_for_lmp_prediction` is `true`. A model directory that exists is **not** enough: artifacts without an implemented loader
are reported as `loader_not_implemented`. The UI sidebar shows "API connected" and "No trained model" as separate indicators.

## Frontend
* **Analyzer:** two text boxes, language selector (English enabled; Hindi listed as "not yet supported" and the Analyze button is
  disabled for it), optional context/legal domain (accepted but not used yet; the response says so), optional baseline request.
  Results show the LMP status, the score and category **only if the backend returned a valid score**, side-by-side texts with
  highlighted spans, expandable findings (category, change type, evidence strength, explanation) and a not-legal-advice notice.
  Without a model it shows: *"Legal Meaning Preservation score unavailable: the trained transformer model has not yet been installed."*
* **Batch analysis:** upload a CSV, see file-level and row-level problems, run, view results, download a CSV.
* **System status:** from `/health` and `/model-info`; metrics appear only if the backend has a saved `metrics.json`.
* **Research info:** the LMP problem, similarity vs preservation, what is implemented vs planned, limitations.

### Batch CSV format (`data/sample/batch_example.csv`; the sentences are invented examples)
```csv
pair_id,original_text,simplified_text,language,context,legal_domain
ex-1,The tenant must pay rent by the fifth day of each month.,The tenant may pay rent by the tenth day of each month.,en,,rental
```
Required: `original_text`, `simplified_text`. Optional: `language` (blank = `en`), `context`, `legal_domain`, and an id column
(`pair_id`, `id` or `row_id`) copied to the results. Headers are case-insensitive; UTF-8 (a BOM is fine).
Limits: 5 MB, 2000 rows, 5000 characters per field. Rows are sent in groups of at most 100.
* **No row is silently dropped.** Invalid rows (blank text, bad/unsupported language such as `hi`, too long) are not sent but
  stay in the results as `invalid_input` with the reason. Results keep `row_number` (data row, header excluded) and `input_id`.
* If the backend goes offline mid-run, rows already answered keep their results; the rest are marked `request_failed` / `not_sent`.
* The downloaded CSV has `status`, `error_code`, `error_message`, `lmp_status`, `predicted_lmp_score` (empty unless a real score
  exists), `preservation_category`, `finding_types`, `findings`, and baseline score columns. Text cells starting with `= + - @` get a
  leading apostrophe so spreadsheets do not execute them as formulas.

## API (summary)
| Endpoint | Purpose |
|---|---|
| `GET /health` | status of model, explanations and baselines, each with a reason |
| `POST /predict` | one pair -> score/category only if a model exists, findings, highlights, optional baselines, warnings |
| `POST /batch-predict` | `{"items": [...]}`, max 100; per-item errors are reported per item |
| `GET /model-info` | backbone, model status, languages, thresholds, baselines, saved evaluation metrics (if any) |

Unavailable score: `lmp_status: "unavailable"`, `predicted_lmp_score: null`, `preservation_category: null`, plus
`lmp_unavailable_reason` (`weights_missing`, `manifest_invalid`, `loader_not_implemented`, `load_error`). Model output outside
[1, 10] or NaN is an error (HTTP 500), never clipped. Errors are `{"error": {"code", "message", ...}}` without stack traces or
paths: `422 validation_error | unsupported_language | input_too_long | batch_too_large`, `413 payload_too_large`,
`500 inference_failed | internal_error`.

Installing a trained model later: `app/inference.py::load_predictor` (artifact contract documented there) or
`create_app(predictor=...)`. Baseline artifacts (`tfidf.pkl` + `tfidf_meta.json`) come from `app.models.baselines.save_tfidf`;
**they are pickles, so only load files you created yourself.**

## Tests
```bash
python -m pytest -q                 # interpreter without Streamlit: 411 passed, 2 skipped (UI tests module + real-model test)
.venv/bin/python -m pytest -q       # with Streamlit installed (requirements-ui.txt): 426 passed, 1 skipped
```
| Group | Files | What it is |
|---|---|---|
| Core | `test_scoring`, `test_dataset`, `test_baselines` | 20 |
| Explanations | `test_explanations`, `test_explanations_dataset` | 159 |
| Backend, no model | `test_api` | 75 |
| Backend, **mock** model | `test_api_mock_model` (marker `mock_model`) | 27, uses `MockPredictor`, a fake |
| Backend, toy baselines | `test_api_baselines` (marker `toy_fixture`) | 16, artifact fitted on invented sentences |
| Backend, real model | `test_api_real_model` (marker `real_model`) | 1, **skipped**: no trained model exists |
| Frontend client | `test_frontend_client` | 61: mocked HTTP; real backend in-process; real uvicorn over HTTP (`live_server`) |
| Frontend batch/CSV | `test_frontend_batch` | 38: mocked HTTP, real backend in-process, one live-server test |
| Frontend display logic | `test_frontend_presenters` | 15 |
| Frontend UI (AppTest) | `test_frontend_app` | 15: 13 with a **mocked** backend, 2 against a real backend; skipped without Streamlit |

Mocked and fixture tests prove how the code handles each state. They do **not** show that a real trained model works.
`python -m pytest -m live_server` runs only the tests that start a real local uvicorn process (offline-safe).

## Troubleshooting
* **"The backend is not reachable" in the UI:** start `uvicorn app.main:app --port 8000`, or set `JUDGEBERT_API_URL` to where it runs.
  Port already in use: pick another (`--port 8010`) and set `JUDGEBERT_API_URL=http://localhost:8010`.
* **"API connected" but "No trained model":** expected today; see "Backend health is not model availability".
* **`ModuleNotFoundError: frontend` / `app`:** run both commands from the repository root.
* **`streamlit: command not found`:** `pip install -r requirements-ui.txt` inside your activated venv.
* **CSV upload rejected:** the message names the problem (missing columns, not UTF-8, too many rows). Re-save as "CSV UTF-8".
* **A row shows `invalid_input`:** the `error_message` column says why; fix that row and re-upload.
* **Hindi text:** deliberately unsupported until Hindi rules, data and tests exist.
* **Behind a proxy / remote backend:** `JUDGEBERT_API_URL` must start with `http://` or `https://`.

## Repository map
`app/` backend (`main`, `routes`, `service`, `schemas`, `inference`, `baseline_service`, `config`, `errors`, `scoring`, `data/`, `models/`, `explanations/`) ·
`frontend/` (`streamlit_app`, `api_client`, `schema_checks`, `presenters`, `batch`) · `tests/` · `docs/` · `data/sample/` ·
`.streamlit/config.toml` · `requirements*.txt` · `pyproject.toml`.

## Data and licensing
No dataset is committed. Only licence-checked public text may be added; `data/raw/` is git-ignored. Test fixtures and
`data/sample/batch_example.csv` are invented illustrations, not research data. A project LICENSE has not been chosen yet.

## Train the XLM-R LMP regressor

The transformer pipeline is implemented, but a trained model is not included. Training requires real, adjudicated human ratings; do not use synthetic/LLM-generated ratings as ground truth.

```bash
pip install -e '.[ml]'
python scripts/train_lmp.py --data path/to/adjudicated_pairs.csv --epochs 4
```

The CSV must follow `data/annotation_template.csv` and contain `adjudicated_rating` (1–10), `source_document_id`, and genuine `human_written` or `expert_reviewed` pairs. The script excludes synthetic pairs, creates train/validation/test partitions grouped by source document, selects the best epoch on validation RMSE, evaluates once on the held-out test partition, and saves the model under `models/lmp/`. The API loads it at startup when PyTorch and Transformers are installed.
