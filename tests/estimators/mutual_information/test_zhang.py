"""Explicit Zhang mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    ZhangMIEstimator,
    ZhangCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.2022458674074699),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 3.0056146685186733),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 1.2022458674074699),
    ],
)
def test_zhang_mi(data_x, data_y, base, expected):
    """Test the zhang mutual information estimator."""
    est = ZhangMIEstimator(data_x, data_y, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.5770780163555852),
    ],
)
def test_zhang_cmi(data_x, data_y, cond, base, expected):
    """Test the zhang conditional mutual information estimator."""
    est = ZhangCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
