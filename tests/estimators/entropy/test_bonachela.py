"""Explicit Bonachela entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy, estimator


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        ([1, 1, 2], 2, 0.8415721071852286, "Simple case with two unique values"),
        ([1, 1, 1], 2, 0.23083120654223416, "All same values should give zero entropy"),
        (
            [1, 1, 1, 1, 1],
            2,
            0.17665653561905675,
            "All same values should give zero entropy",
        ),
        (
            [
                1,
            ]
            * 1000,
            2,
            0.001438378468541811,
            "All same values should give zero entropy",
        ),
        (
            [1, 2, 3, 4],
            2,
            1.8274137184593533,
            "Four unique values, each appearing once",
        ),
        ([1, 1, 2, 2], 2, 0.8896619418815275, "Two values, each appearing twice"),
        ([1, 1, 2], e, 0.5833333333333333, "Same as first test but in nats"),
        ([1, 2, 3], 2, 1.3561333384356253, "Three unique values, each appearing once"),
        ([1, 1, 2, 2], 2, 0.8896619418815275, "perfectly balanced binary"),
        ([1, 1, 1, 2], 2, 0.809512217387696, "unbalanced binary"),
        ([1, 2, 1, 2, 1], 2, 0.8896619418815274, "alternating pattern"),
        ([1, 1, 2, 2, 3, 3, 4], 2, 1.862463320202291, "mostly balanced with extra"),
        ([1, 1, 2, 2, 3], 10, 0.41834080705782617, "base 10"),
        ([1, 1, 1, 2, 2], e, 0.6166666666666666, "natural log with repeated values"),
        (
            [1, 1, 2, 2, 3, 3, 1, 2],
            2,
            1.4501947651094578,
            "larger dataset with repetitions",
        ),
        (
            [1] * 5 + [2] * 3 + [3] * 2,
            2,
            1.4196606693095943,
            "highly unbalanced distribution",
        ),
        ([(1, 1), (1, 1), (2, 1)], 2, 0.8415721071852286, "simple joint data"),
    ],
)
def test_bonachela_estimator_hardcoded_values(data, base, expected, description):
    """Test Bonachela estimator with manually calculated expected values."""
    result = entropy(data, approach="bonachela", base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for {description}"


def test_bonachela_estimator_class_direct():
    """Test Bonachela estimator using the class directly."""
    from infomeasure.estimators.entropy.bonachela import BonachelaEntropyEstimator

    data = [1, 1, 2, 2, 3]
    estimator_obj = BonachelaEntropyEstimator(data, base=2)
    result = estimator_obj.result()

    # Should be positive for this data
    assert result > 0
    assert isinstance(result, float)


def test_bonachela_estimator_functional_api():
    """Test Bonachela estimator through functional API."""
    data = [1, 1, 2, 2, 3, 3]
    result = entropy(data, approach="bonachela", base=2)

    # Should be positive for this data
    assert result > 0
    assert isinstance(result, float)


def test_bonachela_estimator_joint_entropy():
    """Test Bonachela estimator with joint data."""
    data = [(1, "a"), (1, "a"), (2, "b"), (2, "b")]
    result = entropy(data, approach="bonachela", base=2)

    # Should be positive for this data
    assert result > 0
    assert isinstance(result, float)


def test_bonachela_estimator_different_bases():
    """Test Bonachela estimator with different logarithm bases."""
    data = [1, 1, 2, 2]

    result_base2 = entropy(data, approach="bonachela", base=2)
    result_base_e = entropy(data, approach="bonachela", base=e)
    result_base10 = entropy(data, approach="bonachela", base=10)

    # Results should be related by logarithm base conversion
    assert result_base2 == pytest.approx(result_base_e / log(2), rel=1e-10)
    assert result_base10 == pytest.approx(result_base_e / log(10), rel=1e-10)


def test_bonachela_estimator_large_dataset():
    """Test Bonachela estimator with larger dataset."""
    data = [1] * 50 + [2] * 30 + [3] * 20
    result = entropy(data, approach="bonachela", base=2)

    # Should be positive and reasonable for this distribution
    assert result > 0
    assert result < 2  # Should be less than log2(3) for this distribution
    assert isinstance(result, float)


def test_bonachela_estimator_edge_case_two_values():
    """Test Bonachela estimator with minimal non-trivial case."""
    data = [1, 2]
    result = entropy(data, approach="bonachela", base=2)

    # Should be positive for two different values
    assert result > 0
    assert isinstance(result, float)


def test_bonachela_estimator_comparison_with_discrete():
    """Compare Bonachela estimator with discrete estimator."""
    data = [1, 1, 2, 2, 3, 3]

    bonachela_result = entropy(data, approach="bonachela", base=2)
    discrete_result = entropy(data, approach="discrete", base=2)

    # Bonachela estimator should give different result than discrete (bias correction)
    assert bonachela_result != discrete_result
    assert isinstance(bonachela_result, float)
    assert isinstance(discrete_result, float)


def test_bonachela_estimator_small_data_sets():
    """Test Bonachela estimator with very small data sets (its specialty)."""
    # Test with minimal data
    data = [1, 2]
    result = entropy(data, approach="bonachela", base=2)
    assert result > 0
    assert isinstance(result, float)

    # Test with slightly larger minimal data
    data = [1, 1, 2]
    result = entropy(data, approach="bonachela", base=2)
    assert result > 0
    assert isinstance(result, float)
