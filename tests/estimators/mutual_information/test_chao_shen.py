"""Explicit ChaoShen mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    ChaoShenMIEstimator,
    ChaoShenCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0666666666666667),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 5.030519462082247),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 1.0666666666666667),
    ],
)
def test_chao_shen_mi(data_x, data_y, base, expected):
    """Test the chao_shen mutual information estimator."""
    est = ChaoShenMIEstimator(data_x, data_y, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 1.7382172311250148),
    ],
)
def test_chao_shen_cmi(data_x, data_y, cond, base, expected):
    """Test the chao_shen conditional mutual information estimator."""
    est = ChaoShenCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
