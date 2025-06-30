"""Explicit Grassberger transfer entropy estimator tests."""

import pytest
from numpy import e, log, nan, isnan

from tests.conftest import discrete_random_variables_condition
from infomeasure.estimators.transfer_entropy import (
    GrassbergerTEEstimator,
    GrassbergerCTEEstimator,
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
def test_grassberger_te(source, dest, base, expected):
    """Test the grassberger transfer entropy estimator."""
    est = GrassbergerTEEstimator(source, dest, base=base)
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
def test_grassberger_cte(source, dest, cond, base, expected):
    """Test the grassberger conditional transfer entropy estimator."""
    est = GrassbergerCTEEstimator(source, dest, cond=cond, base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "rng_int,high,expected_te,expected_cte",
    [
        (1, 30, 1.260433430591283, 1.3884888887201203),
        (1, 100, 1.5322266994509872, 1.9446421161562217),
    ],
)
def test_grassberger_discrete_shifted(rng_int, high, expected_te, expected_cte):
    """Test the grassberger (conditional) transfer entropy estimator with random data."""
    source, dest, cond = discrete_random_variables_condition(
        rng_int, low=0, high=high, length=100
    )
    est = GrassbergerTEEstimator(source, dest, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_te)
    else:
        assert res == pytest.approx(expected_te, rel=1e-10)

    est = GrassbergerCTEEstimator(source, dest, cond=cond, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_cte)
    else:
        assert res == pytest.approx(expected_cte, rel=1e-10)
