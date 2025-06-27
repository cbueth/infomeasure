"""Explicit Chao Wang Jost entropy estimator tests."""

import pytest
from numpy import e, log
from numpy import sum as np_sum
from scipy.special import digamma

from infomeasure import entropy, estimator


def chao_wang_jost_expected(data, base=2):
    """Calculate expected Chao Wang Jost entropy for given data."""
    import numpy as np

    # Get unique values and counts
    unique_vals, counts = np.unique(data, return_counts=True)
    N = len(data)

    # Calculate singletons (f1) and doubletons (f2)
    f1 = np_sum(counts == 1)
    f2 = np_sum(counts == 2)

    # Calculate parameter A
    if f2 > 0:
        A = 2 * f2 / ((N - 1) * f1 + 2 * f2)
    elif f1 > 0:
        A = 2 / ((N - 1) * (f1 - 1) + 2)
    else:
        A = 1

    # First part of the formula: sum over observed counts
    # Using digamma(N) - digamma(h_i) = sum_{k=h_i}^{N-1} 1/k
    cwj = 0.0
    for count in counts:
        if 1 <= count <= N - 1:
            cwj += count / N * (digamma(N) - digamma(count))

    # Second part: correction term when A != 1
    if A != 1 and f1 > 0:
        # Calculate sum_{r=1}^{N-1} (1/r) * (1-A)^r
        p2 = sum(1 / r * (1 - A) ** r for r in range(1, N))
        correction = f1 / N * (1 - A) ** (1 - N) * (-np.log(A) - p2)
        cwj += correction

    # Convert to the desired base
    if base != "e":
        cwj /= log(base)

    return cwj


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
def test_chao_wang_jost_entropy_basic(data, base):
    """Test the Chao Wang Jost entropy estimator with basic cases."""
    result = entropy(data, approach="chao_wang_jost", base=base)
    assert isinstance(result, float)
    assert result >= 0.0  # Chao Wang Jost entropy should be non-negative


@pytest.mark.parametrize(
    "data,base",
    [
        ([1, 1, 1, 1, 1], 2),  # All same values
        ([1, 0, 1, 0], 2),  # Two different values
        ([1, 2, 3, 4, 5], 2),  # All different values
        ([1, 1, 2, 2, 3], "e"),  # Mixed counts with natural log
        ([1, 1, 2, 2, 2, 3], 2),  # With doubletons
    ],
)
def test_chao_wang_jost_entropy_expected_values(data, base):
    """Test the Chao Wang Jost entropy estimator with calculated expected values."""
    expected = chao_wang_jost_expected(data, base)
    result = entropy(data, approach="chao_wang_jost", base=base)
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
def test_chao_wang_jost_joint_entropy(data, base):
    """Test the Chao Wang Jost joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="chao_wang_jost", base=base)
    result = est.result()
    assert isinstance(result, float)
    assert result >= 0.0  # Chao Wang Jost entropy should be non-negative


@pytest.mark.parametrize("length", [1, 2, 5, 10])
@pytest.mark.parametrize("base", [2, 10, e])
def test_chao_wang_jost_entropy_uniform(length, base):
    """Test the Chao Wang Jost entropy estimator with a uniform distribution."""
    data = list(range(length))
    result = entropy(data, approach="chao_wang_jost", base=base)
    assert isinstance(result, float)
    assert result >= 0.0

    # Compare with expected calculation
    expected = chao_wang_jost_expected(data, base)
    assert result == pytest.approx(expected, rel=1e-10)


def test_chao_wang_jost_vs_discrete():
    """Test that Chao Wang Jost gives different results than discrete entropy."""
    data = [1, 2, 3, 4, 5]  # All singletons - should show bias correction

    cwj_result = entropy(data, approach="chao_wang_jost", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)

    # They should be different (Chao Wang Jost has bias correction)
    assert cwj_result != discrete_result


def test_chao_wang_jost_single_value():
    """Test Chao Wang Jost entropy with single repeated value."""
    data = [5] * 10
    result = entropy(data, approach="chao_wang_jost", base=2)

    # Single value should give 0 entropy
    assert result == pytest.approx(0.0, abs=1e-10)


def test_chao_wang_jost_shorthand():
    """Test that the 'cwj' shorthand works."""
    data = [1, 1, 2, 2, 3]

    result_full = entropy(data, approach="chao_wang_jost", base=2)
    result_short = entropy(data, approach="cwj", base=2)

    assert result_full == result_short


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        ([1, 1, 2], 2, 1.3333333333333333, "simple dataset with counts [2,1]"),
        (
            [0, 0, 1, 1, 2, 2],
            "e",
            1.2833333333333332,
            "uniform counts [2,2,2] with base e",
        ),
        ([1, 1, 1, 1], 2, 0.0, "single value repeated 4 times"),
        ([1, 1, 2, 2], 10, 0.3619120682527097, "two equal groups with base 10"),
        ([1, 2, 2, 3, 3, 3], 2, 1.8083524887820899, "three different count groups"),
        ([1, 2, 3, 4, 5], "e", 2.921221627254949, "uniform distribution 5 elements"),
        ([1, 1, 1, 2], 2, 1.021908987296349, "skewed distribution [3,1]"),
        # 10 additional test cases with varied input data
        ([1, 2, 3, 3, 4, 4, 4], 2, 2.4214827619001236, "mixed counts with tripleton"),
        (
            [1, 1, 2, 2, 3, 3, 4, 4, 5],
            "e",
            1.8402605433347434,
            "mostly doubletons with singleton",
        ),
        (
            [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            10,
            0.6425029693586234,
            "increasing count pattern",
        ),
        (
            [1, 1, 1, 1, 2, 2, 3],
            2,
            1.6783979097232513,
            "dominant group with smaller groups",
        ),
        (
            [1, 2, 3, 4, 5, 6, 7, 8],
            "e",
            3.7901580509810775,
            "all singletons medium size",
        ),
        # 5 larger datasets
        (
            [1] * 20 + [2] * 15 + [3] * 10 + [4] * 5 + list(range(5, 15)),
            2,
            3.2578342091243995,
            "large mixed frequency dataset",
        ),
        (
            [1] * 30 + [2] * 25 + [3] * 20 + [4] * 15 + [5] * 10 + [6] * 5,
            "e",
            1.6865670696999848,
            "large decreasing frequency pattern",
        ),
        (list(range(1, 51)), 2, 10.448232371444393, "large all singletons 50 elements"),
        (
            [1] * 40 + [2] * 30 + [3] * 20 + [4] * 10 + [5, 6, 7, 8, 9],
            10,
            0.6830166355194756,
            "large dataset with mixed patterns",
        ),
        (
            [1, 2] * 25 + [3, 4] * 15 + [5, 6] * 10 + list(range(7, 17)),
            "e",
            2.2844208901120115,
            "large complex pattern with repeats",
        ),
    ],
)
def test_chao_wang_jost_hardcoded_values(data, base, expected, description):
    """Test Chao Wang Jost entropy with hardcoded expected values.

    This test uses manually calculated expected values for specific datasets
    to ensure the implementation is mathematically correct.
    """
    result = entropy(data, approach="chao_wang_jost", base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for {description}"


def test_chao_wang_jost_with_doubletons():
    """Test Chao Wang Jost entropy with data containing doubletons."""
    data = [1, 1, 2, 2, 3, 4, 5]  # f1=3 (singletons: 3,4,5), f2=2 (doubletons: 1,2)
    result = entropy(data, approach="chao_wang_jost", base=2)

    # Should handle the case with both singletons and doubletons
    assert isinstance(result, float)
    assert result > 0.0


def test_chao_wang_jost_no_singletons_doubletons():
    """Test Chao Wang Jost entropy when f1=f2=0 (A=1 case)."""
    data = [1, 1, 1, 2, 2, 2, 3, 3, 3]  # All counts >= 3, so f1=f2=0
    result = entropy(data, approach="chao_wang_jost", base=2)

    # Should handle the A=1 case
    assert isinstance(result, float)
    assert result > 0.0


def test_chao_wang_jost_large_dataset():
    """Test Chao Wang Jost entropy with a larger dataset."""
    import numpy as np

    # Create a dataset with known structure
    data = [1] * 50 + [2] * 30 + [3] * 20 + list(range(4, 14))  # Mix of frequencies

    result = entropy(data, approach="chao_wang_jost", base=2)
    expected = chao_wang_jost_expected(data, base=2)

    assert result == pytest.approx(expected, rel=1e-10)
    assert result > 0.0


def test_chao_wang_jost_local_values():
    """Test that local values raise TheoreticalInconsistencyError."""
    from infomeasure.utils.exceptions import TheoreticalInconsistencyError

    data = [1, 1, 2, 3, 3]
    est = estimator(data, measure="entropy", approach="chao_wang_jost", base=2)

    with pytest.raises(
        TheoreticalInconsistencyError,
        match="Local values are not implemented for Chao Wang Jost estimator",
    ):
        est.local_vals()
