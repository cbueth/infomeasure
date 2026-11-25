"""Explicit Renyi entropy tests."""

import pytest
from numpy import inf

from infomeasure import cross_entropy
from infomeasure.estimators.entropy import (
    RenyiEntropyEstimator,
    DiscreteEntropyEstimator,
)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
def test_renyi_entropy(k, alpha, default_rng):
    """Test the Renyi entropy estimator by design."""
    data = default_rng.normal(0, 10, 1000)
    est = RenyiEntropyEstimator(data, k=k, alpha=alpha)
    if alpha == 1:
        est_discrete = DiscreteEntropyEstimator(data.astype(int))
        assert pytest.approx(est.result(), rel=0.1) == est_discrete.result()
    est.result()


@pytest.mark.parametrize(
    "data,k,alpha,expected",
    [
        ([1, 0, 1, 1, 1, 4, 23, 6, 1, -4, -3], 4, 1.0, 2.522422772),
        ([1, 2, 1, 2, 1, 2, 1, 2], 4, 1.1, 2.015828887),
        ([[0, 0], [1, 3], [2, 2], [3, 1], [4, 0]], 1, 1, 5.8842423067),
        ([[0, 0], [1, 3], [2, 2], [3, 1], [4, 0]], 2, 0.9, 5.297410751),
    ],
)
def test_renyi_entropy_explicit(data, k, alpha, expected):
    """Test the Renyi entropy estimator with specific values."""
    est = RenyiEntropyEstimator(data, k=k, alpha=alpha, base=2)
    assert est.result() == pytest.approx(expected)


def test_renyi_cross_entropy_k1():
    """Test Renyi cross-entropy with k=1 and an explicit evaluation set."""
    data_p = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]
    data_q = [0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]
    result = cross_entropy(data_p, data_q, approach="renyi", k=1, alpha=1.5)
    assert result == pytest.approx(1.8474302989689735)


@pytest.mark.parametrize("k", [0, -1, -10, None])
def test_renyi_entropy_invalid_k(k, default_rng):
    """Test the Renyi entropy estimator with invalid k."""
    data = list(range(10))
    with pytest.raises(ValueError):
        RenyiEntropyEstimator(data, k=k, alpha=1)


@pytest.mark.parametrize("alpha", [0, -1, -10, None])
def test_renyi_entropy_invalid_alpha(alpha, default_rng):
    """Test the Renyi entropy estimator with invalid alpha."""
    data = list(range(10))
    with pytest.raises(ValueError):
        RenyiEntropyEstimator(data, k=1, alpha=alpha)
