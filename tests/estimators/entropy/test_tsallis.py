"""Explicit Tsallis entropy tests."""

import pytest
from numpy import inf

from infomeasure import cross_entropy
from infomeasure.estimators.entropy import (
    TsallisEntropyEstimator,
    DiscreteEntropyEstimator,
)


@pytest.mark.parametrize("k", [1, 2, 5, 10])
@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_tsallis_entropy(k, q, default_rng):
    """Test the discrete entropy estimator."""
    data = default_rng.normal(0, 10, 1000)
    est = TsallisEntropyEstimator(data, k=k, q=q)
    if q == 1:
        est_discrete = DiscreteEntropyEstimator(data.astype(int))
        assert pytest.approx(est.result(), rel=0.1) == est_discrete.result()
    est.result()


@pytest.mark.parametrize(
    "data,k,q,expected",
    [
        ([1, 0, 1, 1, 1, 4, 23, 6, 1, -4, -3], 4, 1.0, 2.522422772),
        ([1, 2, 1, 2, 1, 2, 1, 2], 4, 1.1, 1.3040405910),
        ([[0, 0], [1, 3], [2, 2], [3, 1], [4, 0]], 1, 1, 5.8842423067),
        ([[0, 0], [1, 3], [2, 2], [3, 1], [4, 0]], 2, 0.9, 4.43670072),
    ],
)
def test_tsallis_entropy_explicit(data, k, q, expected):
    """Test the Tsallis entropy estimator with specific values."""
    est = TsallisEntropyEstimator(data, k=k, q=q, base=2)
    assert est.result() == pytest.approx(expected)


def test_tsallis_cross_entropy_k1():
    """Test Tsallis cross-entropy with k=1 and an explicit evaluation set."""
    data_p = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]
    data_q = [0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]
    result = cross_entropy(data_p, data_q, approach="tsallis", k=1, q=1.5)
    assert result == pytest.approx(1.2059175397628703)


@pytest.mark.parametrize("k", [0, -1, -10, None])
def test_tsallis_entropy_invalid_k(k, default_rng):
    """Test the discrete entropy estimator with invalid k."""
    data = list(range(10))
    with pytest.raises(ValueError):
        TsallisEntropyEstimator(data, k=k, q=1)


@pytest.mark.parametrize("q", [0, -1, -10, None])
def test_tsallis_entropy_invalid_q(q, default_rng):
    """Test the discrete entropy estimator with invalid q."""
    data = list(range(10))
    with pytest.raises(ValueError):
        TsallisEntropyEstimator(data, k=1, q=q)
