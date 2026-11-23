"""Baseline tests using a TOY artifact fitted on four invented sentence pairs written in this file.
It exercises artifact loading and validation only; its numbers have no research meaning."""
import json
import pickle

import pytest
from app.baseline_service import BaselineRegistry
from app.models.baselines import TFIDF_FILE, TFIDF_META, TfidfBaseline, load_tfidf, save_tfidf
from app.schemas import PredictResponse
from conftest import PAIR, make_settings

pytestmark = pytest.mark.toy_fixture
O = ["The tenant must pay rent by the fifth day.", "The seller shall deliver the goods within ten days.",
     "Payment is due within thirty days unless otherwise agreed.", "The employee may take leave after notice."]
S = ["The tenant must pay rent by the fifth day.", "The seller shall deliver goods within ten days.",
     "Gardens are lovely in spring.", "Birds fly south in winter."]
META = {"n_train": 4, "languages": ["en"], "mapping": "linear", "fit_scope": "train_only",
        "description": "toy fixture, invented sentences"}


@pytest.fixture
def toy_dir(tmp_path):
    d = tmp_path / "baselines"
    save_tfidf(TfidfBaseline().fit(O, S, [10, 9, 1, 1]), d, META)
    return d


def test_available_baseline_scores(toy_dir, make_client):
    c = make_client()
    d = PredictResponse.model_validate(c.post("/predict", json={**PAIR, "include_baselines": True}).json())
    t = d.baseline_scores.scores["tfidf"]
    assert t.status == "available" and 1.0 <= t.score <= 10.0 and 0.0 <= t.raw_similarity <= 1.0
    assert d.baseline_scores.scores["multilingual_embedding"].status == "unavailable"
    assert d.predicted_lmp_score is None  # baselines never stand in for the LMP score


def test_identical_scores_higher_than_unrelated_on_the_toy_fit(toy_dir, make_client):
    c = make_client()
    ident = c.post("/predict", json={"original_text": O[0], "simplified_text": O[0], "include_baselines": True}).json()
    unrel = c.post("/predict", json={"original_text": O[0], "simplified_text": "Birds fly south in winter.",
                                     "include_baselines": True}).json()
    a, b = ident["baseline_scores"]["scores"]["tfidf"], unrel["baseline_scores"]["scores"]["tfidf"]
    assert a["raw_similarity"] == pytest.approx(1.0) and a["score"] > b["score"]


def test_health_and_model_info_describe_the_artifact(toy_dir, make_client):
    c = make_client()
    assert c.get("/health").json()["baselines"]["tfidf"]["available"] is True
    info = c.get("/model-info").json()["baselines"]["tfidf"]
    assert info["available"] is True and info["metadata"]["fit_scope"] == "train_only" and info["metadata"]["n_train"] == 4
    assert str(toy_dir) not in json.dumps(info)


def test_batch_includes_baselines(toy_dir, make_client):
    d = make_client().post("/batch-predict", json={"items": [PAIR, PAIR], "include_baselines": True}).json()
    assert all(r["result"]["baseline_scores"]["scores"]["tfidf"]["status"] == "available" for r in d["results"])


def test_language_not_covered(tmp_path, make_client):
    d = tmp_path / "baselines"
    save_tfidf(TfidfBaseline().fit(O, S, [10, 9, 1, 1]), d, {**META, "languages": ["hi"]})
    t = make_client().post("/predict", json={**PAIR, "include_baselines": True}).json()["baseline_scores"]["scores"]["tfidf"]
    assert t["status"] == "unavailable" and t["reason"] == "language_not_covered" and t["score"] is None


def _write(d, pkl=None, meta=None):
    d.mkdir(parents=True, exist_ok=True)
    if pkl is not None:
        (d / TFIDF_FILE).write_bytes(pkl)
    if meta is not None:
        (d / TFIDF_META).write_text(meta if isinstance(meta, str) else json.dumps(meta))


def _fitted_bytes():
    return pickle.dumps(TfidfBaseline().fit(O, S, [10, 9, 1, 1]))


@pytest.mark.parametrize("label,pkl,meta,reason", [
    ("corrupt pickle", b"garbage", META, "artifact_invalid"),
    ("not a baseline", pickle.dumps({"a": 1}), META, "artifact_invalid"),
    ("unfitted baseline", pickle.dumps(TfidfBaseline()), META, "artifact_invalid"),
    ("corrupt meta", "fit", "{nope", "artifact_invalid"),
    ("meta missing keys", "fit", {"n_train": 4}, "artifact_invalid"),
    ("not train-only", "fit", {**META, "fit_scope": "all_data"}, "artifact_invalid"),
    ("zero train rows", "fit", {**META, "n_train": 0}, "artifact_invalid"),
    ("pickle only", "fit", None, "artifact_missing"),
    ("meta only", None, META, "artifact_missing"),
])
def test_bad_artifacts_are_rejected_without_crashing(tmp_path, make_client, label, pkl, meta, reason):
    _write(tmp_path / "baselines", _fitted_bytes() if pkl == "fit" else pkl, meta)
    c = make_client()
    t = c.get("/health").json()["baselines"]["tfidf"]
    assert t["available"] is False and t["reason"] == reason, label
    d = c.post("/predict", json={**PAIR, "include_baselines": True}).json()
    assert d["baseline_scores"]["scores"]["tfidf"]["status"] == "unavailable"
    assert str(tmp_path) not in json.dumps(d) and "Traceback" not in json.dumps(d)


def test_scoring_failure_degrades_to_unavailable(toy_dir, tmp_path, make_client):
    reg = BaselineRegistry.load(make_settings(tmp_path))
    reg._tfidf = object()  # breaks scoring at request time
    d = make_client(baselines=reg).post("/predict", json={**PAIR, "include_baselines": True})
    assert d.status_code == 200
    t = d.json()["baseline_scores"]["scores"]["tfidf"]
    assert t["status"] == "unavailable" and t["reason"] == "baseline_error"


def test_save_and_load_validation(tmp_path):
    with pytest.raises(ValueError):
        save_tfidf(TfidfBaseline(), tmp_path / "x", META)  # unfitted
    fitted = TfidfBaseline().fit(O, S, [10, 9, 1, 1])
    for bad in ({**META, "fit_scope": "all_data"}, {**META, "n_train": 0}, {**META, "languages": []}, {"n_train": 4}):
        with pytest.raises(ValueError):
            save_tfidf(fitted, tmp_path / "x", bad)
    save_tfidf(fitted, tmp_path / "ok", META)
    model, meta = load_tfidf(tmp_path / "ok")
    assert meta["fit_scope"] == "train_only" and model.predict([O[0]], [O[0]])[0] > model.predict([O[0]], [S[2]])[0]
    with pytest.raises(FileNotFoundError):
        load_tfidf(tmp_path / "absent")
