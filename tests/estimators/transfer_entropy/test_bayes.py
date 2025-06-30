"""Explicit Bayes transfer entropy estimator tests."""

import pytest
from numpy import e, log, nan, isnan

from tests.conftest import discrete_random_variables_condition
from infomeasure.estimators.transfer_entropy import (
    BayesTEEstimator,
    BayesCTEEstimator,
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
def test_bayes_te(source, dest, base, expected):
    """Test the bayes transfer entropy estimator."""
    est = BayesTEEstimator(source, dest, alpha="jeffrey", base=base)
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
def test_bayes_cte(source, dest, cond, base, expected):
    """Test the bayes conditional transfer entropy estimator."""
    est = BayesCTEEstimator(source, dest, cond=cond, alpha="jeffrey", base=base)
    res = est.result()
    if isnan(expected):
        assert isnan(res)
    else:
        assert res == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "rng_int,high,expected_te,expected_cte",
    [
        (1, 30, 1.5372724970331386, 1.6130398153941448),
        (1, 100, 1.7731997159942745, 1.7999568982579204),
    ],
)
def test_bayes_discrete_shifted(rng_int, high, expected_te, expected_cte):
    """Test the bayes (conditional) transfer entropy estimator with random data."""
    source, dest, cond = discrete_random_variables_condition(
        rng_int, low=0, high=high, length=100
    )
    est = BayesTEEstimator(source, dest, alpha=0.5, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_te)
    else:
        assert res == pytest.approx(expected_te, rel=1e-10)

    est = BayesCTEEstimator(source, dest, alpha=0.5, cond=cond, base=2)
    res = est.result()
    if isnan(res):
        assert isnan(expected_cte)
    else:
        assert res == pytest.approx(expected_cte, rel=1e-10)
