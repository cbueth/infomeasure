"""Explicit ChaoWangJost transfer entropy estimator tests."""

import pytest
from numpy import e, log, nan, isnan

from tests.conftest import discrete_random_variables_condition
from infomeasure.estimators.transfer_entropy import (
    ChaoWangJostTEEstimator,
    ChaoWangJostCTEEstimator,
)


@pytest.mark.parametrize(
    "source,dest,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 0.0),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 0.0),
    ],
)
def test_chao_wang_jost_te(source, dest, base, expected):
    """Test the chao_wang_jost transfer entropy estimator."""
    est = ChaoWangJostTEEstimator(source, dest, base=base)
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
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.0),
    ],
)
def test_chao_wang_jost_cte(source, dest, cond, base, expected):
    """Test the chao_wang_jost conditional transfer entropy estimator."""
    est = ChaoWangJostCTEEstimator(source, dest, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "rng_int,high,expected_te,expected_cte",
    [
        (1, 30, 0.8416099602786642, 1.0423071444251928),
        (1, 100, 0.7202592766413451, 1.795902568738807),
    ],
)
def test_chao_wang_jost_discrete_shifted(rng_int, high, expected_te, expected_cte):
    """Test the chao_wang_jost (conditional) transfer entropy estimator with random data."""
    source, dest, cond = discrete_random_variables_condition(
        rng_int, low=0, high=high, length=100
    )
    est = ChaoWangJostTEEstimator(source, dest, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_te)
    else:
        assert res == pytest.approx(expected_te, rel=1e-10)

    est = ChaoWangJostCTEEstimator(source, dest, cond=cond, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_cte)
    else:
        assert res == pytest.approx(expected_cte, rel=1e-10)
