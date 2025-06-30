"""Explicit Ansb transfer entropy estimator tests."""

import pytest
from numpy import nan, isnan

from infomeasure.estimators.transfer_entropy import (
    AnsbTEEstimator,
    AnsbCTEEstimator,
)


@pytest.mark.parametrize(
    "source,dest,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, nan),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 0.0),
    ],
)
def test_ansb_te(source, dest, base, expected):
    """Test the ansb transfer entropy estimator."""
    est = AnsbTEEstimator(source, dest, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "source,dest,cond,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, nan),
    ],
)
def test_ansb_cte(source, dest, cond, base, expected):
    """Test the ansb conditional transfer entropy estimator."""
    est = AnsbCTEEstimator(source, dest, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
