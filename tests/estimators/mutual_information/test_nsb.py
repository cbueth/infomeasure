"""Explicit Nsb mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    NsbMIEstimator,
    NsbCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, nan),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 0.8709397317797066),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, nan),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 0.8709397317797066),
    ],
)
def test_nsb_mi(data_x, data_y, base, expected):
    """Test the nsb mutual information estimator."""
    est = NsbMIEstimator(data_x, data_y, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data_x,data_y,cond,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, nan),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, nan),
    ],
)
def test_nsb_cmi(data_x, data_y, cond, base, expected):
    """Test the nsb conditional mutual information estimator."""
    est = NsbCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
