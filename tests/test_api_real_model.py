"""Tests that need a REAL trained model. They are skipped unless a model with an implemented loader is installed,
which is NOT the case yet: transformer inference is not integrated and no model has been trained."""
import pytest
from app.config import Settings
from app.inference import load_predictor
from fastapi.testclient import TestClient
from app.main import create_app
from conftest import PAIR

_predictor, _status = load_predictor(Settings.from_env())
pytestmark = [pytest.mark.real_model,
              pytest.mark.skipif(_predictor is None, reason=f"no real trained model available ({_status.reason})")]


def test_real_model_scores_are_bounded_and_categorised():
    with TestClient(create_app()) as c:
        d = c.post("/predict", json=PAIR).json()
        assert d["lmp_status"] == "available" and 1.0 <= d["predicted_lmp_score"] <= 10.0
        assert d["preservation_category"] in {"Preserved", "Partially Preserved", "Not Preserved"}
        assert d["model_version"] and c.get("/health").json()["ready_for_lmp_prediction"] is True
