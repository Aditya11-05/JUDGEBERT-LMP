import numpy as np
import pytest
from app.models.baselines import TfidfBaseline
from app.models.evaluate import regression_metrics

O = ["The tenant must pay rent by the fifth day.", "The seller shall deliver the goods within ten days.",
     "Payment is due within thirty days unless otherwise agreed.", "The employee may take leave after notice."]
S = ["The tenant must pay rent by the fifth day.", "The seller shall deliver goods within ten days.",
     "Gardens are lovely in spring.", "Birds fly south in winter."]
R = [10, 9, 1, 1]


def test_identical_scores_higher_than_unrelated():
    m = TfidfBaseline().fit(O, S, R)
    p = m.predict(O, S)
    assert p[0] > p[2] and p[1] > p[3]
    assert ((p >= 1) & (p <= 10)).all()


def test_raw_similarity_identical_is_one():
    m = TfidfBaseline().fit(O, S, R)
    assert np.isclose(m.raw_similarity([O[0]], [O[0]])[0], 1.0)


def test_unfitted_raises_and_bad_mapping():
    with pytest.raises(RuntimeError):
        TfidfBaseline().predict(O, S)
    with pytest.raises(ValueError):
        TfidfBaseline(mapping="x")


def test_isotonic_in_range():
    p = TfidfBaseline("isotonic").fit(O, S, R).predict(O, S)
    assert ((p >= 1) & (p <= 10)).all()


def test_metrics_values():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 5])
    assert m["rmse"] == pytest.approx(0.5) and m["mae"] == pytest.approx(0.25)
    assert m["pearson"] > 0.9 and m["spearman"] == pytest.approx(1.0)
    assert regression_metrics([1, 2, 3], [5, 5, 5])["pearson"] is None
