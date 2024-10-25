"""Explicit Kozachenko-Leonenko entropy estimator tests."""

import pytest
from numpy import inf

from infomeasure.measures.entropy import KozachenkoLeonenkoEntropyEstimator


@pytest.mark.parametrize("data_len", [100, 1000])
@pytest.mark.parametrize("noise_level", [0, 1e-5])
@pytest.mark.parametrize("minkowski_p", [1.0, 1.5, inf])
@pytest.mark.parametrize("k", [1, 4, 10])
def test_kl_entropy(data_len, noise_level, minkowski_p, k, default_rng):
    """Test the discrete entropy estimator."""
    data = default_rng.integers(0, 10, data_len)
    est = KozachenkoLeonenkoEntropyEstimator(
        data, k=k, noise_level=noise_level, minkowski_p=minkowski_p
    )
    est.results()


# invalid values
@pytest.mark.parametrize(
    "noise_level,minkowski_p,k,match",
    [
        (-1, 1.0, 1, "noise level must be non-negative"),  # noise_level < 0
        (-1e-4, 1.0, 1, "noise level must be non-negative"),  # noise_level < 0
        (0, 0.999, 1, "Minkowski power parameter must be positive"),  # minkowski_p < 1
        (0, 0, 1, "Minkowski power parameter must be positive"),  # minkowski_p < 1
        (0, -inf, 1, "Minkowski power parameter must be positive"),  # minkowski_p < 1
        (0, 1.0, 0, "The number of nearest neighbors"),  # k < 1
        (0, 1.0, 0.5, "The number of nearest neighbors"),  # k < 1
        (0, 1.0, -1, "The number of nearest neighbors"),  # k < 1
        (1, 1.0, 1.0, "The number of nearest neighbors"),  # type(k) != int
        (1, 1.0, 4.0, "The number of nearest neighbors"),  # type(k) != int
    ],
)
def test_invalid_values(noise_level, minkowski_p, k, match, default_rng):
    """Test for invalid values."""
    data = default_rng.integers(0, 10, 100)
    with pytest.raises(ValueError, match=match):
        KozachenkoLeonenkoEntropyEstimator(
            data, k=k, noise_level=noise_level, minkowski_p=minkowski_p
        ).results()
