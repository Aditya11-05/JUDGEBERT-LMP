import math
import pytest
from app.scoring import bounded_score, categorize


@pytest.mark.parametrize("z", [-1e6, -50, -1, 0, 1, 50, 1e6])
def test_bounded(z):
    assert 1.0 <= bounded_score(z) <= 10.0


def test_midpoint_and_monotone():
    assert math.isclose(bounded_score(0), 5.5)
    assert bounded_score(-1) < bounded_score(0) < bounded_score(1)


def test_categories():
    assert categorize(10) == "Preserved"
    assert categorize(7) == "Preserved"
    assert categorize(5) == "Partially Preserved"
    assert categorize(1) == "Not Preserved"
    assert categorize(6.5, preserved_min=6, partial_min=3) == "Preserved"


def test_category_rejects_out_of_range_and_bad_thresholds():
    with pytest.raises(ValueError):
        categorize(11)
    with pytest.raises(ValueError):
        categorize(5, preserved_min=4, partial_min=5)
