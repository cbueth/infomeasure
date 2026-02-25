"""Explicit Bayesian entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy, estimator


@pytest.mark.parametrize(
    "data,alpha,base,expected,description",
    [
        # Simple cases with alpha=1.0 (Laplace smoothing)
        ([1, 1, 1, 1, 1], 1.0, 2, 0.0, "uniform data, alpha=1.0"),
        ([1, 0, 1, 0], 1.0, 2, 1.0, "binary data, alpha=1.0"),
        ([1, 0, 1, 0], "laplace", 2, 1.0, "binary data, alpha='laplace'"),
        # Test with alpha=0.5 (Jeffrey's prior)
        ([1, 0, 1, 0], 0.5, 2, 1.0, "binary data, alpha=0.5"),
        ([1, 0, 1, 0], "Jeffrey", 2, 1.0, "binary data, alpha='Jeffrey''"),
        # Test with different alpha values
        ([1, 2, 3], 1.0, 2, 1.5849625007211563, "three symbols, alpha=1.0"),
        ([1, 2, 3], 0.5, 2, 1.5849625007211563, "three symbols, alpha=0.5"),
        # Test with base e
        ([1, 0, 1, 0], 1.0, "e", 0.6931471805599453, "binary data, alpha=1.0, base=e"),
        # Test with base 10
        ([1, 0, 1, 0], 1.0, 10, 0.30102999566398114, "binary data, alpha=1.0, base=10"),
    ],
)
def test_bayes_entropy_hardcoded_values(data, alpha, base, expected, description):
    """Test Bayesian entropy estimator with manually calculated expected values."""
    result = entropy(data, approach="bayes", alpha=alpha, base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for: {description}"


@pytest.mark.parametrize(
    "data,alpha,K,base,expected,description",
    [
        # Test with explicit K parameter
        ([1, 0, 1, 0], 1.0, 3, 2, 1.0477649325740983, "binary data with K=3"),
        ([1, 0, 1, 0], 1.0, 5, 2, 1.0566416671474375, "binary data with K=5"),
    ],
)
def test_bayes_entropy_with_K_parameter(data, alpha, K, base, expected, description):
    """Test Bayesian entropy estimator with explicit K parameter."""
    result = entropy(data, approach="bayes", alpha=alpha, K=K, base=base)
    assert result == pytest.approx(expected, rel=1e-10), f"Failed for: {description}"


@pytest.mark.parametrize(
    "data,alpha,base,expected",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 1.0, 2, 0.0),  # uniform joint data
        (
            ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3]),
            1.0,
            2,
            2.4922031072054134,  # approximate expected value for joint entropy
        ),
        (
            ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3]),
            "jeffrey",
            2,
            2.4497381910452884,  # approximate expected value for joint entropy
        ),
        (
            ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3]),
            "laplace",
            2,
            2.4922031072054134,  # approximate expected value for joint entropy
        ),
        (
            ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3]),
            "sch-grass",
            2,
            2.402393185063893,  # approximate expected value for joint entropy
        ),
        (
            ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3]),
            "min-max",
            2,
            2.455489294962795,  # approximate expected value for joint entropy
        ),
    ],
)
def test_bayes_joint_entropy(data, alpha, base, expected):
    """Test the Bayesian joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="bayes", alpha=alpha, base=base)
    result = est.result()
    # assert result >= 0, "Entropy should be non-negative"
    assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(
    "data_p, data_q, alpha",
    [
        ([0, 1], [0, 1], 1.0),
        ([0, 1], [1, 0], 1.0),
        ([0, 1, 2], [0, 1, 2], 1.0),
        ([0, 1, 2], [2, 1, 0], 1.0),
        ([0, 1], [0, 1], 0.5),  # Jeffrey's prior
    ],
)
def test_bayes_cross_entropy(data_p, data_q, alpha):
    """Test the Bayesian cross-entropy estimator."""
    result = entropy(data_p, data_q, approach="bayes", alpha=alpha)
    assert result >= 0, "Cross-entropy should be non-negative"


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("base", [2, 10, e])
def test_bayes_entropy_different_alphas_and_bases(alpha, base):
    """Test Bayesian entropy with different alpha values and bases."""
    data = [0, 1, 0, 1, 2, 2]  # Simple test data
    result = entropy(data, approach="bayes", alpha=alpha, base=base)
    assert result >= 0, f"Entropy should be non-negative for alpha={alpha}, base={base}"


def test_bayes_entropy_properties():
    """Test mathematical properties of Bayesian entropy."""
    # Test that entropy is non-negative
    data = [1, 2, 3, 1, 2, 3]
    result = entropy(data, approach="bayes", alpha=1.0)
    assert result >= 0, "Entropy should be non-negative"

    # Test that adding more diversity increases entropy
    uniform_data = [1, 2, 3, 4]
    skewed_data = [1, 1, 1, 2]
    uniform_entropy = entropy(uniform_data, approach="bayes", alpha=1.0)
    skewed_entropy = entropy(skewed_data, approach="bayes", alpha=1.0)
    assert uniform_entropy > skewed_entropy, (
        "More uniform data should have higher entropy"
    )


def test_bayes_entropy_edge_cases():
    """Test edge cases for Bayesian entropy."""
    # Single unique value - should be 0 (or very close due to floating point precision)
    single_value_data = [1, 1, 1, 1]
    result = entropy(single_value_data, approach="bayes", alpha=1.0)
    assert result == pytest.approx(0.0, abs=1e-15), (
        "Entropy should be 0 for single value data"
    )

    # Empty intersection in cross-entropy (should be handled gracefully)
    data_p = [1, 1, 1]
    data_q = [2, 2, 2]
    result = entropy(data_p, data_q, approach="bayes", alpha=1.0)
    assert result >= 0, "Cross-entropy should handle empty intersection gracefully"


def test_bayes_estimator_class_direct():
    """Test using the BayesEntropyEstimator class directly."""
    from infomeasure.estimators.entropy.bayes import BayesEntropyEstimator

    data = [0, 1, 0, 1]
    est = BayesEntropyEstimator(data, alpha=1.0, base=2)
    result = est.result()
    assert result >= 0, "Direct estimator usage should work"


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0, 10.0])
def test_bayes_entropy_alpha_sensitivity(alpha):
    """Test that different alpha values produce reasonable results."""
    data = [0, 1, 2, 0, 1, 2]
    result = entropy(data, approach="bayes", alpha=alpha)
    assert result >= 0, f"Entropy should be non-negative for alpha={alpha}"
    assert result < 10, (
        f"Entropy should be reasonable for alpha={alpha}"
    )  # Sanity check
