"""Explicit Chao-Shen entropy estimator tests."""

import pytest
from numpy import e, log
from numpy import sum as np_sum

from infomeasure import entropy, estimator


def chao_shen_expected(data, base=2):
    """Calculate expected Chao-Shen entropy for given data."""
    import numpy as np

    # Get unique values and counts
    unique_vals, counts = np.unique(data, return_counts=True)
    N = len(data)
    # Number of singletons
    f1 = np_sum(counts == 1)
    if f1 == N:
        f1 -= 1  # Avoid C=0

    # Estimated coverage
    C = 1 - f1 / N
    pa = C * counts / len(data)  # Coverage adjusted empirical frequencies
    la = 1 - (1 - pa) ** N  # Probability to see a bin (species) in the sample

    # Chao-Shen (2003) entropy estimator
    H = -np_sum(pa * np.log(pa) / la)

    if base != "e":
        H /= log(base)

    return H


@pytest.mark.parametrize(
    "data,base",
    [
        ([1, 1, 1, 1, 1], 2),  # All same values
        ([1, 0, 1, 0], 2),  # Two different values
        (["a", 0, "a", 0], 2),  # Mixed types
        ([1, 2, 3, 4, 5], 2),  # All different values (all singletons)
        ([1, 2, 3, 4, 5], 10),  # Different base
        ([1, 2, 3, 4, 5], "e"),  # Natural log base
    ],
)
def test_chao_shen_entropy_basic(data, base):
    """Test the Chao-Shen entropy estimator with basic cases."""
    result = entropy(data, approach="chao_shen", base=base)
    assert isinstance(result, float)
    assert result >= 0.0  # Chao-Shen entropy should be non-negative


@pytest.mark.parametrize(
    "data,base",
    [
        ([1, 1, 1, 1, 1], 2),  # All same values
        ([1, 0, 1, 0], 2),  # Two different values
        ([1, 2, 3, 4, 5], 2),  # All different values
        ([1, 1, 2, 2, 3], "e"),  # Mixed counts with natural log
    ],
)
def test_chao_shen_entropy_expected_values(data, base):
    """Test the Chao-Shen entropy estimator with calculated expected values."""
    expected = chao_shen_expected(data, base)
    result = entropy(data, approach="chao_shen", base=base)
    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data,base",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2),  # Same joint values
        (
            ([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]),
            2,
        ),  # Different joint values
        (([1, 2, 3] * 10, [1, 2, 3] * 10), 10),  # Larger dataset
    ],
)
def test_chao_shen_joint_entropy(data, base):
    """Test the Chao-Shen joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="chao_shen", base=base)
    result = est.result()
    assert isinstance(result, float)
    assert result >= 0.0  # Chao-Shen entropy should be non-negative


@pytest.mark.parametrize("length", [1, 2, 5, 10])
@pytest.mark.parametrize("base", [2, 10, e])
def test_chao_shen_entropy_uniform(length, base):
    """Test the Chao-Shen entropy estimator with a uniform distribution."""
    data = list(range(length))
    result = entropy(data, approach="chao_shen", base=base)
    assert isinstance(result, float)
    assert result >= 0.0

    # Compare with expected calculation
    expected = chao_shen_expected(data, base)
    assert result == pytest.approx(expected, rel=1e-10)


def test_chao_shen_vs_discrete():
    """Test that Chao-Shen gives different results than discrete entropy."""
    data = [1, 2, 3, 4, 5]  # All singletons - should show bias correction

    chao_shen_result = entropy(data, approach="chao_shen", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)

    # They should be different (Chao-Shen has bias correction)
    assert chao_shen_result != discrete_result


def test_chao_shen_single_value():
    """Test Chao-Shen entropy with single repeated value."""
    data = [5] * 10
    result = entropy(data, approach="chao_shen", base=2)

    # Single value should give 0 entropy
    assert result == pytest.approx(0.0, abs=1e-10)


def test_chao_shen_shorthand():
    """Test that the 'cs' shorthand works."""
    data = [1, 1, 2, 2, 3]

    result_full = entropy(data, approach="chao_shen", base=2)
    result_short = entropy(data, approach="cs", base=2)

    assert result_full == result_short


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        ([1, 1, 2], 2, 1.53826957554, "simple dataset with counts [2,1]"),
        (
            [0, 0, 1, 1, 2, 2],
            "e",
            1.204343396148,
            "uniform counts [2,2,2] with base e",
        ),
        ([1, 1, 1, 1], 2, 0.0, "single value repeated 4 times"),
        ([1, 1, 2, 2], 10, 0.32109866204157994, "two equal groups with base 10"),
        ([1, 2, 2, 3, 3, 3], 2, 1.813923751653, "three different count groups"),
        ([1, 2, 3, 4, 5], "e", 3.4868903818, "uniform distribution 5 elements"),
        ([1, 1, 1, 2], 2, 1.2872696988, "skewed distribution [3,1]"),
    ],
)
def test_chao_shen_hardcoded_values(data, base, expected, description):
    """Test Chao-Shen entropy with hardcoded expected values.

    This test uses manually calculated expected values for specific datasets
    to ensure the implementation is mathematically correct.
    """
    result = entropy(data, approach="chao_shen", base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for {description}"


def test_chao_shen_coverage_edge_case():
    """Test Chao-Shen entropy when all observations are singletons."""
    data = [1, 2, 3, 4, 5]  # All singletons, f1 = N
    result = entropy(data, approach="chao_shen", base=2)

    # Should handle the f1 = N case by setting f1 = N - 1
    assert isinstance(result, float)
    assert result > 0.0


def test_chao_shen_large_dataset():
    """Test Chao-Shen entropy with a larger dataset."""
    import numpy as np

    # Create a dataset with known structure
    data = [1] * 50 + [2] * 30 + [3] * 20 + list(range(4, 14))  # Mix of frequencies

    result = entropy(data, approach="chao_shen", base=2)
    expected = chao_shen_expected(data, base=2)

    assert result == pytest.approx(2.410138672, rel=1e-5)
    assert result > 0.0
