"""Explicit Grassberger entropy estimator tests."""

import pytest
from numpy import e, log
from scipy.special import digamma

from infomeasure import entropy, estimator
from infomeasure.utils.exceptions import TheoreticalInconsistencyError


def grassberger_expected(counts, N, base=2):
    """Calculate expected Grassberger entropy for given counts."""
    entropy_val = 0.0
    for n_i in counts:
        prob = n_i / N
        term = log(N) - digamma(n_i) - ((-1) ** n_i) / (n_i + 1)
        entropy_val += prob * term

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
def test_grassberger_entropy_basic(data, base):
    """Test the Grassberger entropy estimator with basic cases."""
    result = entropy(data, approach="grassberger", base=base)
    assert isinstance(result, float)
    # Note: Grassberger can produce negative values due to bias correction


@pytest.mark.parametrize(
    "data,base",
    [
        ([1, 1, 1, 1, 1], 2),  # counts=[5], N=5
        ([1, 0, 1, 0], 2),  # counts=[2, 2], N=4
        ([1, 2, 3, 4, 5], 2),  # counts=[1, 1, 1, 1, 1], N=5
    ],
)
def test_grassberger_entropy_expected_values(data, base):
    """Test the Grassberger entropy estimator with calculated expected values."""
    from collections import Counter

    counts = list(Counter(data).values())
    N = len(data)
    expected = grassberger_expected(counts, N, base)

    result = entropy(data, approach="grassberger", base=base)
    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data,base",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2),  # Same joint values
        (
            ([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]),
            2,
        ),  # Different joint values
        (([1, 2, 3 - 2] * 100, [1, 2, 3 - 2] * 100), 10),  # Larger dataset
    ],
)
def test_grassberger_joint_entropy(data, base):
    """Test the Grassberger joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="grassberger", base=base)
    result = est.result()
    assert isinstance(result, float)
    # Note: Grassberger can produce negative values due to bias correction

    # Test that local values can be computed
    local_vals = est.local_vals()
    assert len(local_vals) == len(data[0])


@pytest.mark.parametrize("length", [1, 2, 5, 10])
@pytest.mark.parametrize("base", [2, 10, e])
def test_grassberger_entropy_uniform(length, base):
    """Test the Grassberger entropy estimator with a uniform distribution."""
    data = list(range(length))
    result = entropy(data, approach="grassberger", base=base)
    assert isinstance(result, float)
    # Note: Grassberger can produce negative values due to bias correction

    # For uniform distribution, all counts are 1
    counts = [1] * length
    expected = grassberger_expected(counts, length, base)
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
def test_grassberger_cross_entropy(data_p, data_q):
    """Test that Grassberger cross-entropy raises appropriate TheoreticalInconsistencyError."""
    with pytest.raises(
        TheoreticalInconsistencyError,
        match="Cross-entropy is not implemented for Grassberger estimator",
    ):
        entropy(data_p, data_q, approach="grassberger")


def test_grassberger_vs_discrete():
    """Test that Grassberger gives different results than discrete entropy."""
    data = [1, 2, 3, 4, 5]

    grassberger_result = entropy(data, approach="grassberger", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)

    # They should be different (Grassberger has bias correction)
    assert grassberger_result != discrete_result


def test_grassberger_local_values():
    """Test that local values are computed correctly."""
    data = [1, 1, 2, 2, 3]
    est = estimator(data, measure="entropy", approach="grassberger", base=2)

    local_vals = est.local_vals()
    assert len(local_vals) == len(data)
    assert all(isinstance(val, float) for val in local_vals)


def test_grassberger_empty_intersection():
    """Test that Grassberger cross-entropy raises TheoreticalInconsistencyError even with no common support."""
    data_p = [1, 2, 3]
    data_q = [4, 5, 6]

    with pytest.raises(
        TheoreticalInconsistencyError,
        match="Cross-entropy is not implemented for Grassberger estimator",
    ):
        entropy(data_p, data_q, approach="grassberger")


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        ([1, 1, 2], 2, 1.3757622595782166, "simple dataset with counts [2,1]"),
        (
            [0, 0, 1, 1, 2, 2],
            "e",
            1.0356418007962545,
            "uniform counts [2,2,2] with base e",
        ),
        ([1, 1, 1, 1], 2, -0.10073373919735826, "single value repeated 4 times"),
        ([1, 1, 2, 2], 10, 0.2736822602584788, "two equal groups with base 10"),
        ([1, 2, 2, 3, 3, 3], 2, 1.9950510682325175, "three different count groups"),
        ([1, 2, 3, 4, 5], "e", 2.6866535773356333, "uniform distribution 5 elements"),
        ([1, 1, 1, 2], 2, 1.6605564565545845, "skewed distribution [3,1]"),
    ],
)
def test_grassberger_hardcoded_values(data, base, expected, description):
    """Test Grassberger entropy with hardcoded expected values.

    This test uses manually calculated expected values for specific datasets
    to ensure the implementation is mathematically correct.
    """
    result = entropy(data, approach="grassberger", base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for {description}"


def test_grassberger_single_value():
    """Test Grassberger entropy with single repeated value."""
    data = [5] * 10
    result = entropy(data, approach="grassberger", base=2)

    # Grassberger can produce negative values due to bias correction
    # This is mathematically correct behavior
    assert isinstance(result, float)
    assert (
        abs(result) < 0.1
    )  # Should be small in absolute value for uniform single value
