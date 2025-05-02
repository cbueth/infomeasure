"""Explicit Kozachenko-Leonenko entropy estimator tests."""

import pytest
from numpy import inf

from infomeasure.estimators.entropy import KozachenkoLeonenkoEntropyEstimator


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
    est.result()


@pytest.mark.parametrize(
    "data,minkowski_p,k,expected",
    [
        ([1, 0, 1, 1, 1, 4, 23, 6, 1], inf, 4, 1.941190518),
        ([1, 0, 1, 0, 1, 0], inf, 3, 2.130111115363),
        ([1, 0, 1, 0, 1, 0], inf, 1, 0.0),
        ([1, 2, 3, 4, 5], inf, 2, 2.96291962762),
        ([1, 2, 3, 4, 5], inf, 3, 3.0755571074),
        ([1, 2, 3, 4, 5], 2, 2, 2.9629196276),
        ([1, 2, 3, 4, 5], 2, 3, 3.0755571074),
        (
            [
                [1, 4, 2],
                [3, 2, 1],
                [1, 2, 3],
                [2, 3, 1],
                [3, 1, 2],
                [2, 1, 3],
                [1, 3, 2],
                [2, 3, 1],
                [10, 20, 30],
                [30, 20, 10],
                [-7, 5, 1],
            ],
            inf,
            4,
            9.4310220245,
        ),
        (
            [
                [1, 4, 2],
                [3, 2, 1],
                [1, 2, 3],
                [2, 3, 1],
                [3, 1, 2],
                [2, 1, 3],
                [1, 3, 2],
                [2, 3, 1],
                [10, 20, 30],
                [30, 20, 10],
                [-7, 5, 1],
            ],
            2,
            4,
            9.5907392025,
        ),
    ],
)
def test_kl_entropy_explicit(data, minkowski_p, k, expected):
    """Test the Kozachenko-Leonenko entropy estimator with specific values."""
    est = KozachenkoLeonenkoEntropyEstimator(
        data, k=k, minkowski_p=minkowski_p, noise_level=0, base=2
    )
    assert est.global_val() == pytest.approx(expected)


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
        ).result()
