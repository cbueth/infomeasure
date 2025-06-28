"""Explicit Bayes mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    BayesMIEstimator,
    BayesCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 2.321928094887362),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 1.0),
    ],
)
def test_bayes_mi(data_x, data_y, base, expected):
    """Test the bayes mutual information estimator."""
    est = BayesMIEstimator(data_x, data_y, alpha="jeffrey", base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.3627376714674164),
    ],
)
def test_bayes_cmi(data_x, data_y, cond, base, expected):
    """Test the bayes conditional mutual information estimator."""
    est = BayesCMIEstimator(data_x, data_y, cond=cond, alpha="jeffrey", base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
