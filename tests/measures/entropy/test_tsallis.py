"""Explicit Tsallis entropy tests."""

import pytest

from infomeasure.measures.entropy import TsallisEntropyEstimator


@pytest.mark.parametrize("data_len", [100, 1000])
@pytest.mark.parametrize("k", [1, 2, 5, 10])
@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.0, 3.0])
def test_tsallis_entropy(data_len, k, q, default_rng):
    """Test the discrete entropy estimator."""
    data = default_rng.integers(0, 10, data_len)
    est = TsallisEntropyEstimator(data, k=k, q=q)
    # if q == 1:
    #     est_discrete = DiscreteEntropyEstimator(data)
    #     assert est.results() == est_discrete.results()
    est.results()


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
