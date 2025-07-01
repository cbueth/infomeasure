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


def test_ansb_entropy_with_k_parameter():
    """Test ANSB entropy with explicit K parameter."""
    from infomeasure.estimators.entropy.ansb import AnsbEntropyEstimator
    from numpy import euler_gamma, log
    from scipy.special import digamma

    # Test data with observed K=4, but specify larger K
    data = [1, 1, 2, 3, 4]  # N=5, observed K=4
    N = len(data)

    # Test with K larger than observed
    K_specified = 10
    est = AnsbEntropyEstimator(data, K=K_specified, base="e")
    result = est.result()

    # Manual calculation with specified K
    coincidences = N - K_specified  # 5 - 10 = -5
    # Since coincidences <= 0, this should result in NaN
    assert isnan(result), "ANSB should return NaN when K > N"


def test_ansb_entropy_k_parameter_functional_api():
    """Test ANSB entropy K parameter through functional API."""
    # Test data with coincidences
    data = [1, 1, 2, 3, 4, 5]  # N=6, observed K=5

    # Test with default K (observed)
    result_default = entropy(data, approach="ansb", base="e")

    # Test with explicit K equal to observed
    result_explicit = entropy(data, approach="ansb", K=5, base="e")

    # Should be identical
    assert result_default == pytest.approx(result_explicit, rel=1e-10)


@pytest.mark.parametrize(
    "data,K_specified,expected_coincidences,description",
    [
        ([1, 1, 2, 3], 3, 1, "K equals observed, one coincidence"),
        ([1, 1, 2, 3], 2, 2, "K smaller than observed, more coincidences"),
        ([1, 1, 2, 3], 5, -1, "K larger than N, negative coincidences"),
        ([1, 1, 1, 1], 1, 3, "All same values, K=1"),
        ([1, 2, 3, 4], 6, -2, "No observed coincidences, K > N"),
    ],
)
def test_ansb_entropy_k_parameter_coincidences(data, K_specified, expected_coincidences, description):
    """Test ANSB entropy with different K values and their effect on coincidences."""
    from infomeasure.estimators.entropy.ansb import AnsbEntropyEstimator

    N = len(data)
    actual_coincidences = N - K_specified
    assert actual_coincidences == expected_coincidences, f"Coincidence calculation error for {description}"

    est = AnsbEntropyEstimator(data, K=K_specified, base="e")
    result = est.result()

    if expected_coincidences <= 0:
        assert isnan(result), f"Should return NaN for {description} (non-positive coincidences)"
    else:
        assert not isnan(result), f"Should return finite value for {description}"
        assert result > 0, f"Entropy should be positive for {description}"


def test_ansb_entropy_k_parameter_manual_calculation():
    """Test ANSB entropy with K parameter using manual calculation."""
    from numpy import euler_gamma, log
    from scipy.special import digamma

    data = [1, 1, 2, 3, 4, 5, 6]  # N=7, observed K=6
    N = len(data)
    K_specified = 4  # Specify smaller K

    # Manual calculation
    coincidences = N - K_specified  # 7 - 4 = 3
    gamma = euler_gamma
    expected = (gamma - log(2)) + 2 * log(N) - digamma(coincidences)

    result = entropy(data, approach="ansb", K=K_specified, base="e")
    assert result == pytest.approx(expected, rel=1e-10)


def test_ansb_entropy_k_parameter_undersampling_warning(caplog):
    """Test undersampling warning with K parameter."""
    data = [1, 1, 2, 3, 4]  # N=5

    # Test with K that makes ratio > undersampled threshold
    K_large = 50  # N/K = 5/50 = 0.1, exactly at default threshold
    entropy(data, approach="ansb", K=K_large, undersampled=0.05)  # Lower threshold

    # Should warn about undersampling
    assert "Data is not sufficiently undersampled" in caplog.text


def test_ansb_entropy_k_parameter_edge_cases():
    """Test ANSB entropy K parameter edge cases."""
    data = [1, 2, 3]  # N=3

    # Test K = N (no coincidences)
    result_k_equals_n = entropy(data, approach="ansb", K=3, base="e")
    assert isnan(result_k_equals_n), "Should return NaN when K=N (no coincidences)"

    # Test K > N (negative coincidences)
    result_k_greater_n = entropy(data, approach="ansb", K=5, base="e")
    assert isnan(result_k_greater_n), "Should return NaN when K>N"

    # Test K = 1 (maximum coincidences)
    result_k_one = entropy(data, approach="ansb", K=1, base="e")
    assert not isnan(result_k_one), "Should work when K=1"
    assert result_k_one > 0, "Should be positive when K=1"


def test_ansb_entropy_k_parameter_base_conversion():
    """Test ANSB entropy K parameter with different bases."""
    data = [1, 1, 2, 3, 4, 5]  # N=6
    K_specified = 4

    # Calculate in different bases
    result_e = entropy(data, approach="ansb", K=K_specified, base="e")
    result_2 = entropy(data, approach="ansb", K=K_specified, base=2)
    result_10 = entropy(data, approach="ansb", K=K_specified, base=10)

    # Check conversion relationships
    assert result_2 == pytest.approx(result_e / log(2), rel=1e-10)
    assert result_10 == pytest.approx(result_e / log(10), rel=1e-10)


def test_ansb_entropy_k_parameter_vs_observed():
    """Test comparison between specified K and observed K."""
    data = [1, 1, 2, 3, 4]  # N=5, observed K=4

    # Default behavior (uses observed K=4)
    result_observed = entropy(data, approach="ansb", base="e")

    # Explicit K=4 (same as observed)
    result_explicit_same = entropy(data, approach="ansb", K=4, base="e")

    # Different K=3 (more coincidences)
    result_explicit_diff = entropy(data, approach="ansb", K=3, base="e")

    # Same K should give same result
    assert result_observed == pytest.approx(result_explicit_same, rel=1e-10)

    # Different K should give different result
    assert result_observed != pytest.approx(result_explicit_diff, rel=1e-10)

    # Smaller K (more coincidences) should generally give different entropy
    assert not isnan(result_explicit_diff), "Should work with K=3"
