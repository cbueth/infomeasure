"""Explicit Renyi entropy tests."""

import pytest

from infomeasure.measures.entropy import RenyiEntropyEstimator


@pytest.mark.parametrize("data_len", [100, 1000])
@pytest.mark.parametrize("k", [1, 2, 5, 10])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_renyi_entropy(data_len, k, alpha, default_rng):
    """Test the discrete entropy estimator."""
    data = default_rng.integers(0, 10, data_len)
    est = RenyiEntropyEstimator(data, k=k, alpha=alpha)
    # if alpha == 1:
    #     est_discrete = KozachenkoLeonenkoEntropyEstimator(data)
    #     assert est.results() == est_discrete.results()
    est.results()


@pytest.mark.parametrize("k", [0, -1, -10, None])
def test_renyi_entropy_invalid_k(k, default_rng):
    """Test the discrete entropy estimator with invalid k."""
    data = list(range(10))
    with pytest.raises(ValueError):
        RenyiEntropyEstimator(data, k=k, alpha=1)


@pytest.mark.parametrize("alpha", [0, -1, -10, None])
def test_renyi_entropy_invalid_alpha(alpha, default_rng):
    """Test the discrete entropy estimator with invalid alpha."""
    data = list(range(10))
    with pytest.raises(ValueError):
        RenyiEntropyEstimator(data, k=1, alpha=alpha)
