"""Explicit ANSB entropy estimator tests."""

import pytest
from numpy import e, log, nan, isnan
from scipy.special import digamma

from infomeasure import entropy, estimator
from infomeasure.utils.exceptions import TheoreticalInconsistencyError


@pytest.mark.parametrize(
    "data,base,expected,description",
    [
        # Test case with coincidences (repeated values)
        ([1, 1, 2, 3, 4], 2, 5.309348544, "data with coincidences"),
        ([1, 1, 1, 2, 2], 2, 3.1453059829, "data with multiple coincidences"),
        ([1, 1, 2, 2, 3, 3], 2, 3.671374794662, "data with pairs"),
        # Test case with no coincidences (should return NaN)
        ([1, 2, 3, 4, 5], 2, nan, "data with no coincidences"),
        ([1, 2, 3], 2, nan, "small data with no coincidences"),
    ],
)
def test_ansb_entropy_basic(data, base, expected, description):
    """Test the ANSB entropy estimator with basic cases."""
    result = entropy(data, approach="ansb", base=base)

    if expected is nan:
        assert isnan(result), f"Expected NaN for {description}, got {result}"
    else:
        # For cases with coincidences, we can't easily predict the exact value
        # but we can check that it's a finite number
        assert not isnan(result), (
            f"Expected finite value for {description}, got {result}"
        )
        assert result == pytest.approx(expected)


def test_ansb_entropy_manual_calculation():
    """Test ANSB entropy with manual calculation for verification."""
    # Data with known coincidences
    data = [1, 1, 2, 3, 4, 5]  # One coincidence (two 1's)
    N = len(data)
    delta = 1  # Number of coincidences

    # Manual calculation: (γ - log(2)) + 2*log(N) - ψ(Δ)
    gamma = 0.5772156649015329  # Euler's gamma
    expected = (gamma - log(2)) + 2 * log(N) - digamma(delta)

    result = entropy(data, approach="ansb", base="e")
    assert result == pytest.approx(expected, rel=1e-10)


def test_ansb_entropy_base_conversion():
    """Test ANSB entropy with different bases."""
    data = [1, 1, 2, 3, 4]  # Data with coincidences

    # Calculate in nats (base e)
    result_e = entropy(data, approach="ansb", base="e")

    # Calculate in bits (base 2)
    result_2 = entropy(data, approach="ansb", base=2)

    # Calculate in base 10
    result_10 = entropy(data, approach="ansb", base=10)

    # Check conversion relationships
    assert result_2 == pytest.approx(result_e / log(2), rel=1e-10)
    assert result_10 == pytest.approx(result_e / log(10), rel=1e-10)


def test_ansb_entropy_no_coincidences():
    """Test ANSB entropy when there are no coincidences."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All unique values

    result = entropy(data, approach="ansb", base=2)
    assert isnan(result), "ANSB should return NaN when there are no coincidences"


def test_ansb_entropy_all_same():
    """Test ANSB entropy when all values are the same."""
    data = [1, 1, 1, 1, 1]  # All same values
    N = len(data)
    delta = N - 1  # Number of coincidences (N-1 for all same)

    # Manual calculation
    gamma = 0.5772156649015329
    expected = (gamma - log(2)) + 2 * log(N) - digamma(delta)

    result = entropy(data, approach="ansb", base="e")
    assert result == pytest.approx(expected, rel=1e-10)


def test_ansb_entropy_undersampled_warning(caplog):
    """Test that ANSB works with undersampled data."""
    # Create undersampled data (many unique values, small sample)
    data = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # K=9, N=10, ratio = 0.9 > 0.1

    # Should still work but might warn about not being sufficiently undersampled
    result = entropy(data, approach="ansb", base=2)
    assert not isnan(result), "ANSB should work even if not optimally undersampled"
    assert "Data is not sufficiently undersampled (N/K =" in caplog.text


@pytest.mark.parametrize(
    "data_p, data_q",
    [
        ([1, 1, 2], [1, 1, 2]),  # Same distributions with coincidences
        ([1, 1, 2], [2, 2, 3]),  # Different distributions with coincidences
    ],
)
def test_ansb_cross_entropy_not_implemented(data_p, data_q):
    """Test that ANSB cross-entropy raises appropriate error."""
    with pytest.raises(TheoreticalInconsistencyError):
        entropy(data_p, data_q, approach="ansb")


def test_ansb_joint_entropy():
    """Test ANSB joint entropy calculation."""
    # Create joint data that will have coincidences when combined into tuples
    data = (
        [1, 1, 2, 2],
        [1, 1, 3, 3],
    )  # This creates tuples: (1,1), (1,1), (2,3), (2,3)
    # So we have coincidences: two (1,1) tuples and two (2,3) tuples

    est = estimator(data, measure="entropy", approach="ansb", base=2)
    result = est.result()

    # Should return a finite positive value
    assert not isnan(result), (
        "Joint entropy should not be NaN for data with coincidences"
    )
    assert result > 0, "Joint entropy should be positive"


def test_ansb_estimator_class():
    """Test using the ANSB estimator class directly."""
    from infomeasure.estimators.entropy.ansb import AnsbEntropyEstimator

    data = [1, 1, 2, 3, 4]
    est = AnsbEntropyEstimator(data, base=2)

    result = est.result()
    assert not isnan(result), "Direct estimator usage should work"
    assert result > 0, "Entropy should be positive"

    # Test that it has the expected methods
    assert hasattr(est, "_simple_entropy")
    assert hasattr(est, "_joint_entropy")
    assert hasattr(est, "_extract_local_values")


def test_ansb_edge_cases():
    """Test ANSB entropy with edge cases."""
    # Single element (no coincidences possible)
    result_single = entropy([1], approach="ansb", base=2)
    assert isnan(result_single), "Single element should return NaN"

    # Two identical elements
    data_two = [1, 1]
    result_two = entropy(data_two, approach="ansb", base="e")

    # Manual calculation for two identical elements
    N = 2
    delta = 1  # One coincidence
    gamma = 0.5772156649015329
    expected_two = (gamma - log(2)) + 2 * log(N) - digamma(delta)

    assert result_two == pytest.approx(expected_two, rel=1e-10)
