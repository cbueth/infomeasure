"""Explicit Shrink (James-Stein) entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy, estimator
from infomeasure.utils.exceptions import TheoreticalInconsistencyError


def shrink_expected(data, base=2):
    """Calculate expected shrink entropy for given data."""
    from collections import Counter

    counts = list(Counter(data).values())
    N = len(data)
    K = len(counts)

    # Maximum likelihood probabilities
    p_ml = [count / N for count in counts]

    # Target probabilities (uniform distribution)
    t = 1.0 / K

    # Calculate lambda (shrinkage parameter)
    if N == 0 or N == 1:
        lambda_shrink = 1.0
    else:
        # Variance of u
        varu = [p * (1.0 - p) / (N - 1) for p in p_ml]

        # Mean squared difference
        msp = sum((p - t) ** 2 for p in p_ml)

        if msp == 0:
            lambda_shrink = 1.0
        else:
            lambda_val = sum(varu) / msp
            # Clamp lambda to [0, 1]
            if lambda_val > 1:
                lambda_shrink = 1.0
            elif lambda_val < 0:
                lambda_shrink = 0.0
            else:
                lambda_shrink = lambda_val

    # Calculate shrinkage probabilities
    p_shrink = [lambda_shrink * t + (1 - lambda_shrink) * p for p in p_ml]

    # Calculate entropy
    entropy_val = -sum(p * log(p) for p in p_shrink)

    if base != "e":
        entropy_val /= log(base)

    return entropy_val


@pytest.mark.parametrize(
    "data,base",
    [
        ([1, 1, 1, 1, 1], 2),  # All same values
        ([1, 0, 1, 0], 2),  # Two different values
        (["a", 0, "a", 0], 2),  # Mixed types
        ([1, 2, 3, 4, 5], 2),  # All different values
        ([1, 2, 3, 4, 5], 10),  # Different base
        ([1, 2, 3, 4, 5], "e"),  # Natural log base
    ],
)
def test_shrink_entropy_basic(data, base):
    """Test the shrink entropy estimator with basic cases."""
    result = entropy(data, approach="shrink", base=base)
    assert isinstance(result, float)
    assert result >= 0  # Entropy should be non-negative


@pytest.mark.parametrize(
    "data,base",
    [
        ([1, 1, 1, 1, 1], 2),  # All same values
        ([1, 0, 1, 0], 2),  # Two different values
        ([1, 2, 3, 4, 5], 2),  # All different values
        ([1, 1, 2, 2, 3], "e"),  # Mixed counts with natural base
    ],
)
def test_shrink_entropy_expected_values(data, base):
    """Test the shrink entropy estimator with calculated expected values."""
    expected = shrink_expected(data, base)
    result = entropy(data, approach="shrink", base=base)
    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data,base",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2),  # Same joint values
        (
            ([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]),
            2,
        ),  # Different joint values
        (([1, 2, 3] * 100, [1, 2, 3] * 100), 10),  # Larger dataset
    ],
)
def test_shrink_joint_entropy(data, base):
    """Test the shrink joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="shrink", base=base)
    result = est.result()
    assert isinstance(result, float)
    assert result >= 0  # Entropy should be non-negative

    # Test that local values can be computed
    local_vals = est.local_vals()
    assert len(local_vals) == len(data[0])


@pytest.mark.parametrize("length", [1, 2, 5, 10])
@pytest.mark.parametrize("base", [2, 10, e])
def test_shrink_entropy_uniform(length, base):
    """Test the shrink entropy estimator with a uniform distribution."""
    data = list(range(length))
    result = entropy(data, approach="shrink", base=base)
    assert isinstance(result, float)
    assert result >= 0  # Entropy should be non-negative

    # For uniform distribution, verify against expected calculation
    expected = shrink_expected(data, base)
    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data_p, data_q",
    [
        ([0, 1], [0, 1]),
        ([0, 1], [1, 0]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [2, 1, 0]),
    ],
)
def test_shrink_cross_entropy(data_p, data_q):
    """Test that shrink cross-entropy raises appropriate TheoreticalInconsistencyError."""
    with pytest.raises(
        TheoreticalInconsistencyError,
        match="Cross-entropy is not implemented for shrinkage estimator",
    ):
        entropy(data_p, data_q, approach="shrink")


def test_shrink_vs_discrete():
    """Test that shrink gives different results than discrete entropy."""
    data = [1, 2, 3, 4, 5, 4, 3, 4, 5]

    shrink_result = entropy(data, approach="shrink", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)
    print(shrink_result, discrete_result)

    # They should be different (shrink has bias correction)
    assert shrink_result != discrete_result


@pytest.mark.parametrize("n_values", [1, 2, 5, 10])
@pytest.mark.parametrize("base", [2, e])
@pytest.mark.parametrize("length", [5, 1000])
def test_shrink_equals_discrete_for_uniform(n_values, base, length):
    """Test that shrink and discrete estimators give same results for uniform distributions."""
    # Create uniform distribution with many samples
    data = list(range(n_values)) * length

    shrink_result = entropy(data, approach="shrink", base=base)
    discrete_result = entropy(data, approach="discrete", base=base)

    # For uniform distributions with large samples, they should be equal
    assert shrink_result == pytest.approx(discrete_result, rel=1e-10)


def test_shrink_local_values():
    """Test that local values are computed correctly."""
    data = [1, 1, 2, 2, 3]
    est = estimator(data, measure="entropy", approach="shrink", base=2)

    local_vals = est.local_vals()
    assert len(local_vals) == len(data)
    assert all(isinstance(val, float) for val in local_vals)
    assert all(val >= 0 for val in local_vals)  # Local values should be non-negative


def test_shrink_empty_intersection():
    """Test that shrink cross-entropy raises TheoreticalInconsistencyError even with no common support."""
    data_p = [1, 2, 3]
    data_q = [4, 5, 6]

    with pytest.raises(
        TheoreticalInconsistencyError,
        match="Cross-entropy is not implemented for shrinkage estimator",
    ):
        entropy(data_p, data_q, approach="shrink")


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        ([1, 1, 2], 2, 1.0000000000000000, "simple dataset with counts [2,1]"),
        (
            [0, 0, 1, 1, 2, 2],
            "e",
            1.0986122886681096,
            "uniform counts [2,2,2] with base e",
        ),
        ([1, 1, 1, 1], 2, 0.0, "single value repeated 4 times"),
        ([1, 1, 2, 2], 10, 0.3010299956639811, "two equal groups with base 10"),
        ([1, 2, 2, 3, 3, 3], 2, 1.5849625007211561, "three different count groups"),
        ([1, 2, 3, 4, 5], "e", 1.6094379124341005, "uniform distribution 5 elements"),
        ([1, 1, 1, 2], 2, 1.0000000000000000, "skewed distribution [3,1]"),
    ],
)
def test_shrink_hardcoded_values(data, base, expected, description):
    """Test shrink entropy with hardcoded expected values.

    This test uses manually calculated expected values for specific datasets
    to ensure the implementation is mathematically correct.
    """
    result = entropy(data, approach="shrink", base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for {description}"


def test_shrink_single_value():
    """Test shrink entropy with single repeated value."""
    data = [5] * 10
    result = entropy(data, approach="shrink", base=2)

    # For a single value, entropy should be 0
    assert result == pytest.approx(0.0, abs=1e-10)


def test_shrink_functional_aliases():
    """Test that both 'shrink' and 'js' functional aliases work."""
    data = [1, 1, 2, 2, 3]

    shrink_result = entropy(data, approach="shrink", base=2)
    js_result = entropy(data, approach="js", base=2)

    assert shrink_result == js_result


def test_shrink_lambda_edge_cases():
    """Test shrink estimator with edge cases for lambda calculation."""
    # Test with N=1 (should use lambda=1)
    data_single = [1]
    result_single = entropy(data_single, approach="shrink", base=2)
    assert result_single == pytest.approx(0.0, abs=1e-10)

    # Test with uniform distribution (lambda should be close to 1)
    data_uniform = [1, 2, 3, 4, 5] * 20  # Large uniform sample
    result_uniform = entropy(data_uniform, approach="shrink", base=2)
    assert isinstance(result_uniform, float)
    assert result_uniform > 0


def test_shrink_vs_maximum_likelihood():
    """Test that shrink estimator shrinks towards uniform distribution."""
    # For small samples, shrink should be closer to uniform than ML
    data = [1, 1, 2]  # Highly skewed small sample

    shrink_result = entropy(data, approach="shrink", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)

    # Shrink should give higher entropy (closer to uniform) for small skewed samples
    assert shrink_result > discrete_result
