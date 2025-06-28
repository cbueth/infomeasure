"""Explicit Grassberger mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    GrassbergerMIEstimator,
    GrassbergerCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.38950877712704973),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 0.9091527894249158),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 3.8760217926087117),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 0.9091527894249158),
    ],
)
def test_grassberger_mi(data_x, data_y, base, expected):
    """Test the grassberger mutual information estimator."""
    est = GrassbergerMIEstimator(data_x, data_y, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 1.057976363318573),
    ],
)
def test_grassberger_cmi(data_x, data_y, cond, base, expected):
    """Test the grassberger conditional mutual information estimator."""
    est = GrassbergerCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
