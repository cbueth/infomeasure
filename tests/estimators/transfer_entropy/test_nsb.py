"""Explicit Nsb transfer entropy estimator tests."""

import pytest
from numpy import e, log, nan, isnan

from infomeasure.estimators.transfer_entropy import (
    NsbTEEstimator,
    NsbCTEEstimator,
)


@pytest.mark.parametrize(
    "source,dest,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, nan),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, nan),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 0.0),
    ],
)
def test_nsb_te(source, dest, base, expected):
    """Test the nsb transfer entropy estimator."""
    est = NsbTEEstimator(source, dest, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "source,dest,cond,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, nan),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, nan),
    ],
)
def test_nsb_cte(source, dest, cond, base, expected):
    """Test the nsb conditional transfer entropy estimator."""
    est = NsbCTEEstimator(source, dest, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)
