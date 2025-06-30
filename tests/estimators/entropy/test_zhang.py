"""Explicit Zhang entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy, estimator
from infomeasure.utils.data import DiscreteData


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        ([1, 1, 2], 2, 1.2022458674074694, "Simple case with two unique values"),
        ([1, 1, 1], 2, 0.0, "All same values should give zero entropy"),
        ([1, 2, 3, 4], 2, 2.644940908296433, "Four unique values, each appearing once"),
        ([1, 1, 2, 2], 2, 1.2022458674074699, "Two values, each appearing twice"),
        ([1, 1, 2], e, 0.8333333333333333, "Same as first test but in nats"),
        ([1, 2, 3], 2, 2.1640425613334453, "Three unique values, each appearing once"),
        ([1, 1, 2, 2], 2, 1.2022458674074699, "perfectly balanced binary"),
        ([1, 1, 1, 2], 2, 1.021908987296349, "unbalanced binary"),
        ([1, 2, 1, 2, 1], 2, 1.1301111153630214, "alternating pattern"),
        ([1, 1, 2, 2, 3, 3, 4], 2, 2.298007100844563, "mostly balanced with extra"),
        ([1, 1, 2, 2, 3], 10, 0.5573445851091731, "base 10"),
        ([1, 1, 1, 2, 2], e, 0.7833333333333334, "natural log with repeated values"),
        (
            [1, 1, 2, 2, 3, 3, 1, 2],
            2,
            1.756996460511202,
            "larger dataset with repetitions",
        ),
        (
            [1] * 5 + [2] * 3 + [3] * 2,
            2,
            1.6407793599951466,
            "highly unbalanced distribution",
        ),
        ([(1, 1), (1, 1), (2, 1)], 2, 1.2022458674074694, "simple joint data"),
    ],
)
def test_zhang_estimator_hardcoded_values(data, base, expected, description):
    """Test Zhang estimator with manually calculated expected values."""
    result = entropy(data, approach="zhang", base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for {description}"


def test_zhang_estimator_class_direct():
    """Test Zhang estimator using the class directly."""
    from infomeasure.estimators.entropy.zhang import ZhangEntropyEstimator

    data = [1, 1, 2, 2, 3]
    estimator_obj = ZhangEntropyEstimator(data, base=2)
    result = estimator_obj.result()

    # Should be positive for this data
    assert result > 0
    assert isinstance(result, float)


def test_zhang_estimator_functional_api():
    """Test Zhang estimator through functional API."""
    data = [1, 1, 2, 2, 3, 3]
    result = entropy(data, approach="zhang", base=2)

    # Should be positive for this data
    assert result > 0
    assert isinstance(result, float)


def test_zhang_estimator_joint_entropy():
    """Test Zhang estimator with joint data."""
    data = [(1, "a"), (1, "a"), (2, "b"), (2, "b")]
    result = entropy(data, approach="zhang", base=2)

    # Should be positive for this data
    assert result > 0
    assert isinstance(result, float)


def test_zhang_estimator_single_value():
    """Test Zhang estimator with single unique value."""
    data = [1, 1, 1, 1]
    result = entropy(data, approach="zhang", base=2)

    # Should be zero for constant data
    assert result == pytest.approx(0.0, abs=1e-10)


def test_zhang_estimator_different_bases():
    """Test Zhang estimator with different logarithm bases."""
    data = [1, 1, 2, 2]

    result_base2 = entropy(data, approach="zhang", base=2)
    result_base_e = entropy(data, approach="zhang", base=e)
    result_base10 = entropy(data, approach="zhang", base=10)

    # Results should be related by logarithm base conversion
    assert result_base2 == pytest.approx(result_base_e / log(2), rel=1e-10)
    assert result_base10 == pytest.approx(result_base_e / log(10), rel=1e-10)


def test_zhang_estimator_large_dataset():
    """Test Zhang estimator with larger dataset."""
    data = [1] * 50 + [2] * 30 + [3] * 20
    result = entropy(data, approach="zhang", base=2)

    # Should be positive and reasonable for this distribution
    assert result > 0
    assert result < 2  # Should be less than log2(3) for this distribution
    assert isinstance(result, float)


def test_zhang_estimator_edge_case_two_values():
    """Test Zhang estimator with minimal non-trivial case."""
    data = [1, 2]
    result = entropy(data, approach="zhang", base=2)

    # Should be positive for two different values
    assert result > 0
    assert isinstance(result, float)


def test_zhang_estimator_comparison_with_discrete():
    """Compare Zhang estimator with discrete estimator."""
    data = [1, 1, 2, 2, 3, 3]

    zhang_result = entropy(data, approach="zhang", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)

    # Zhang estimator should give different result than discrete (bias correction)
    assert zhang_result != discrete_result
    assert isinstance(zhang_result, float)
    assert isinstance(discrete_result, float)


def zhang_entropy_original(data, base="e"):
    """Original Zhang entropy implementation with loops.

    Detailed Timing Analysis
    ========================================
    Size  100: Original 0.000592s, Vectorized 0.000033s, Speedup 18.17x
    Size  200: Original 0.002471s, Vectorized 0.000046s, Speedup 53.89x
    Size  500: Original 0.015711s, Vectorized 0.000134s, Speedup 117.17x
    Size 1000: Original 0.063930s, Vectorized 0.000489s, Speedup 130.75x
    Size 2000: Original 0.259134s, Vectorized 0.002403s, Speedup 107.86x
    Size 5000: Original 1.599501s, Vectorized 0.015616s, Speedup 102.43x

    Using np.random.choice(range(n_unique_values), size=n_samples) with
    n_unique_values = n_samples/10.
    """
    discrete_data = DiscreteData.from_data(data)
    counts = discrete_data.counts
    N = discrete_data.N

    ent = 0.0

    # Iterate over each unique value and its count
    for count in counts:
        # Skip if count is 0 or greater than N-1 (edge case)
        if count == 0 or count >= N:
            continue

        # Calculate the inner sum with product
        t1 = 1.0
        t2 = 0.0

        for k in range(1, N - count + 1):
            t1 *= 1.0 - (count - 1.0) / (N - k)
            t2 += t1 / k

        # Add contribution to entropy
        ent += t2 * (count / N)

    # Convert to the desired base if needed
    if base != "e":
        ent /= log(base)

    return ent


@pytest.mark.parametrize(
    "data_len,fraction_unique",
    [
        (100, 0.1),
        (100, 0.2),
        (100, 0.5),
        (100, 2.0),
        (1000, 0.1),
        (1000, 1.0),
        (2000, 0.01),
    ],
)
@pytest.mark.parametrize("base", ["e", 2])
def test_zhang_estimator_against_original_implementation(
    data_len, fraction_unique, base, default_rng
):
    """Test Zhang estimator against the original implementation."""
    data = default_rng.choice(
        int(data_len * fraction_unique), size=data_len, replace=True
    )
    result = entropy(data, approach="zhang", base=base)
    original_result = zhang_entropy_original(data, base=base)
    assert result == pytest.approx(original_result, rel=1e-10)
