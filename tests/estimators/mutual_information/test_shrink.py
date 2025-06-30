"""Explicit Shrink mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    ShrinkMIEstimator,
    ShrinkCMIEstimator,
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
def test_shrink_mi(data_x, data_y, base, expected):
    """Test the shrink mutual information estimator."""
    est = ShrinkMIEstimator(data_x, data_y, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.3219280948873622),
    ],
)
def test_shrink_cmi(data_x, data_y, cond, base, expected):
    """Test the shrink conditional mutual information estimator."""
    est = ShrinkCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
