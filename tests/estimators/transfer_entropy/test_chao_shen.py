"""Explicit ChaoShen transfer entropy estimator tests."""

import pytest
from numpy import e, log, nan, isnan

from tests.conftest import discrete_random_variables_condition
from infomeasure.estimators.transfer_entropy import (
    ChaoShenTEEstimator,
    ChaoShenCTEEstimator,
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
def test_chao_shen_te(source, dest, base, expected):
    """Test the chao_shen transfer entropy estimator."""
    est = ChaoShenTEEstimator(source, dest, base=base)
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
def test_chao_shen_cte(source, dest, cond, base, expected):
    """Test the chao_shen conditional transfer entropy estimator."""
    est = ChaoShenCTEEstimator(source, dest, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "rng_int,high,expected_te,expected_cte",
    [
        (1, 30, 0.8658655136658213, 1.0289330952232332),
        (1, 100, 0.6140209315813459, 1.5304859625842155),
    ],
)
def test_chao_shen_discrete_shifted(rng_int, high, expected_te, expected_cte):
    """Test the chao_shen (conditional) transfer entropy estimator with random data."""
    source, dest, cond = discrete_random_variables_condition(
        rng_int, low=0, high=high, length=100
    )
    est = ChaoShenTEEstimator(source, dest, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_te)
    else:
        assert res == pytest.approx(expected_te, rel=1e-10)

    est = ChaoShenCTEEstimator(source, dest, cond=cond, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_cte)
    else:
        assert res == pytest.approx(expected_cte, rel=1e-10)
