"""Explicit Renyi entropy tests."""

import pytest

from infomeasure.measures.entropy import RenyiEntropyEstimator, DiscreteEntropyEstimator


@pytest.mark.parametrize("k", [1, 2, 5, 10])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_renyi_entropy(k, alpha, default_rng):
    """Test the Renyi entropy estimator by design."""
    data = default_rng.normal(0, 10, 1000)
    est = RenyiEntropyEstimator(data, k=k, alpha=alpha)
    if alpha == 1:
        est_discrete = DiscreteEntropyEstimator(data.astype(int))
        assert pytest.approx(est.results(), rel=0.1) == est_discrete.results()
    est.results()


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
