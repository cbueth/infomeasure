"""Explicit Ansb mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    AnsbMIEstimator,
    AnsbCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 2.6644076360320263),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 3.222797313664771),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, nan),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 3.222797313664771),
    ],
)
def test_ansb_mi(data_x, data_y, base, expected):
    """Test the ansb mutual information estimator."""
    est = AnsbMIEstimator(data_x, data_y, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, nan),
    ],
)
def test_ansb_cmi(data_x, data_y, cond, base, expected):
    """Test the ansb conditional mutual information estimator."""
    est = AnsbCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
