"""Explicit Bonachela mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    BonachelaMIEstimator,
    BonachelaCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.17665653561905675),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 0.8896619418815275),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 2.2523708291429734),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 0.8896619418815275),
    ],
)
def test_bonachela_mi(data_x, data_y, base, expected):
    """Test the bonachela mutual information estimator."""
    est = BonachelaMIEstimator(data_x, data_y, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data_x,data_y,cond,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.4313363744698637),
    ],
)
def test_bonachela_cmi(data_x, data_y, cond, base, expected):
    """Test the bonachela conditional mutual information estimator."""
    est = BonachelaCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
