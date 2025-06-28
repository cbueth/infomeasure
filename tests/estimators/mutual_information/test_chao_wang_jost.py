"""Explicit ChaoWangJost mutual information estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.mutual_information import (
    ChaoWangJostMIEstimator,
    ChaoWangJostCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.2022458674074692),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 4.214431954978303),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 1.2022458674074692),
    ],
)
def test_chao_wang_jost_mi(data_x, data_y, base, expected):
    """Test the chao_wang_jost mutual information estimator."""
    est = ChaoWangJostMIEstimator(data_x, data_y, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 1.182013703838042),
    ],
)
def test_chao_wang_jost_cmi(data_x, data_y, cond, base, expected):
    """Test the chao_wang_jost conditional mutual information estimator."""
    est = ChaoWangJostCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
