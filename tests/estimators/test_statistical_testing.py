"""Test for the statistical testing functionality."""

import numpy as np
import pytest

from infomeasure.estimators.mixins import StatisticalTestingMixin
from infomeasure.utils.data import StatisticalTestResult


@pytest.mark.parametrize("n_tests", [2, 5, 300])
def test_mutual_information_statistical_test(mi_estimator, n_tests):
    """Test the statistical test for mutual information."""
    # Use data with some correlation to get meaningful p-values and t-scores
    np.random.seed(42)
    data_x = np.random.normal(0, 1, 50)
    data_y = data_x + np.random.normal(0, 0.5, 50)  # Correlated data
    estimator, kwargs = mi_estimator
    estimator = estimator(data_x, data_y, **kwargs)
    result = estimator.statistical_test(n_tests)
    assert isinstance(result, StatisticalTestResult)
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.t_score, float)
    assert result.n_tests == n_tests
    assert result.method in ["permutation_test", "bootstrap"]
    # Test percentiles can be calculated on demand
    percentiles_95 = result.percentile([2.5, 97.5])  # 95% CI
    assert len(percentiles_95) == 2
    if not np.isnan(estimator.global_val()):
        assert percentiles_95[0] <= percentiles_95[1]


@pytest.mark.parametrize("n_tests", [2, 5, 300])
def test_mi_statistical_test_comprehensive(mi_estimator, n_tests):
    """Test comprehensive statistical test for MI with different confidence levels."""
    # Use data with some dependency to get meaningful results
    np.random.seed(123)
    source = np.random.exponential(1, 50)
    dest = source * 0.8 + np.random.normal(0, 0.3, 50)  # Dependent data
    estimator, kwargs = mi_estimator
    estimator = estimator(source, dest, **kwargs)
    result = estimator.statistical_test(n_tests)
    assert isinstance(result, StatisticalTestResult)
    assert isinstance(result.t_score, float)
    assert result.n_tests == n_tests
    assert result.method in ["permutation_test", "bootstrap"]
    # Test multiple confidence levels can be calculated on demand using percentiles
    percentiles_90 = result.percentile([5, 95])  # 90% CI
    percentiles_95 = result.percentile([2.5, 97.5])  # 95% CI
    percentiles_99 = result.percentile([0.5, 99.5])  # 99% CI
    assert len(percentiles_90) == 2
    assert len(percentiles_95) == 2
    assert len(percentiles_99) == 2
    # 99% CI should be wider than 95% CI which should be wider than 90% CI
    if not np.isnan(estimator.global_val()):
        assert (
            (percentiles_99[1] - percentiles_99[0])
            >= (percentiles_95[1] - percentiles_95[0])
            >= (percentiles_90[1] - percentiles_90[0])
        )


@pytest.mark.parametrize("n_tests", [2, 5, 50])
def test_transfer_entropy_statistical_test(te_estimator, n_tests):
    """Test the statistical test for transfer entropy."""
    # Use data with some temporal dependency to get meaningful results
    np.random.seed(456)
    source = np.random.randint(0, 3, 100)
    dest = np.zeros(100, dtype=int)
    # Create some temporal dependency: dest[t] depends on source[t-1]
    for t in range(1, 100):
        if source[t - 1] == 0:
            dest[t] = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            dest[t] = np.random.choice([0, 1], p=[0.3, 0.7])

    estimator, kwargs = te_estimator
    estimator = estimator(source, dest, **kwargs)
    result = estimator.statistical_test(n_tests)
    assert isinstance(result, StatisticalTestResult)
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.t_score, float)
    assert result.n_tests == n_tests
    assert result.method in ["permutation_test", "bootstrap"]
    # Test percentiles can be calculated on demand
    percentiles_95 = result.percentile([2.5, 97.5])  # 95% CI
    assert len(percentiles_95) == 2
    assert percentiles_95[0] <= percentiles_95[1]
    # Test that results are consistent when called multiple times
    result2 = estimator.statistical_test(n_tests)
    assert isinstance(result2, StatisticalTestResult)


@pytest.mark.parametrize("n_tests", [2, 5, 25])
def test_te_statistical_test_comprehensive(te_estimator, n_tests):
    """Test comprehensive statistical test for TE with different methods."""
    # Use data with temporal dependency for meaningful results
    np.random.seed(789)
    source = np.random.randint(0, 2, 100)
    dest = np.zeros(100, dtype=int)
    # Create temporal dependency
    for t in range(1, 100):
        dest[t] = (source[t - 1] + np.random.randint(0, 2)) % 2

    estimator, kwargs = te_estimator
    estimator = estimator(source, dest, **kwargs)

    # Test with permutation test
    result_perm = estimator.statistical_test(n_tests, method="permutation_test")
    assert isinstance(result_perm, StatisticalTestResult)
    assert isinstance(result_perm.t_score, float)
    assert result_perm.method == "permutation_test"
    assert result_perm.n_tests == n_tests

    # Test with bootstrap
    result_boot = estimator.statistical_test(n_tests, method="bootstrap")
    assert isinstance(result_boot, StatisticalTestResult)
    assert isinstance(result_boot.t_score, float)
    assert result_boot.method == "bootstrap"
    assert result_boot.n_tests == n_tests


@pytest.mark.parametrize(
    "observed_value, test_values, p_value, t_score",
    [
        (0.5, [0.1, 0.2, 0.3, 0.4, 0.5], 0.0, 1.2649110640),
        (0.25, [0.1, 0.2, 0.3, 0.4, 0.5], 0.6, -0.316227766),
        (0.3, [0.1, 0.2, 0.3, 0.4, 0.5], 0.4, 0.0),
        (0.35, [0.1, 0.2, 0.3, 0.4, 0.5], 0.4, 0.3162277660),
        (0.1, [0.1, 0.2, 0.3, 0.4, 0.5], 0.8, -1.264911064),
        (0.0, [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, -1.897366596),
        (0.09, [0.1, 0.2, 0.3, 0.4, 0.5], 1.0, -1.328156617),
        (1.0, [-1, 1], 0.0, 0.707106781),
        (0.0, [-2, 0], 0.0, 0.707106781),
        (1.0, [1, 1], 0, np.nan),  # Not enough variance in test_values
        (0.0, [2.0] * 10, 1.0, np.nan),  # Not enough variance in test_values
    ],
)
def test_statistical_test_result(observed_value, test_values, p_value, t_score):
    """Test the comprehensive statistical test result calculation."""
    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )
    assert isinstance(result, StatisticalTestResult)
    assert result.p_value == pytest.approx(p_value, abs=1e-6)
    if not np.isnan(t_score):
        assert result.t_score == pytest.approx(t_score, abs=1e-6)
    else:
        assert np.isnan(result.t_score)

    # Test metadata
    assert result.n_tests == len(test_values)
    assert result.method == "permutation_test"

    # Test percentiles can be calculated on demand
    percentiles_95 = result.percentile([2.5, 97.5])  # 95% CI
    assert len(percentiles_95) == 2
    assert percentiles_95[0] <= percentiles_95[1]  # Lower bound <= upper bound


@pytest.mark.parametrize("n_test_vals", [0, 1])
def test_statistical_test_result_not_enough_test_values(n_test_vals):
    """Test the statistical test result calculation with not enough test values."""
    with pytest.raises(ValueError, match="Not enough test values for statistical test"):
        StatisticalTestingMixin._statistical_test_result(
            1.0, [1] * n_test_vals, n_tests=n_test_vals, method="permutation_test"
        )


def test_statistical_test_result_percentiles():
    """Test percentile calculation with multiple levels."""
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    observed_value = 0.55

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="bootstrap"
    )

    assert isinstance(result, StatisticalTestResult)
    assert result.n_tests == len(test_values)
    assert result.method == "bootstrap"

    # Test multiple percentiles can be calculated on demand
    percentile_levels = [10, 25, 50, 75, 90, 95, 99]
    for p in percentile_levels:
        percentile_value = result.percentile(p)
        assert np.isscalar(percentile_value)

    # Test percentile method with multiple values
    percentiles_95 = result.percentile([2.5, 97.5])  # 95% CI
    assert isinstance(percentiles_95, np.ndarray)
    assert len(percentiles_95) == 2

    # Test that wider confidence intervals are indeed wider
    percentiles_80 = result.percentile([10, 90])  # 80% CI
    percentiles_99 = result.percentile([0.5, 99.5])  # 99% CI
    assert isinstance(percentiles_80, np.ndarray)
    assert isinstance(percentiles_99, np.ndarray)
    assert (
        (percentiles_99[1] - percentiles_99[0])
        >= (percentiles_95[1] - percentiles_95[0])
        >= (percentiles_80[1] - percentiles_80[0])
    )


def test_statistical_test_result_percentile_methods():
    """Test percentile method with different parameters."""
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    observed_value = 0.3

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )

    # Test single percentile
    median = result.percentile(50)
    assert np.isscalar(median)
    assert median == 0.3  # Median of [0.1, 0.2, 0.3, 0.4, 0.5]

    # Test multiple percentiles
    quartiles = result.percentile([25, 75])
    assert isinstance(quartiles, np.ndarray)
    assert len(quartiles) == 2
    assert quartiles[0] <= quartiles[1]

    # Test different methods
    p95_linear = result.percentile(95, method="linear")
    p95_nearest = result.percentile(95, method="nearest")
    assert np.isscalar(p95_linear)
    assert np.isscalar(p95_nearest)


def test_percentile_comprehensive():
    """Test comprehensive percentile functionality."""
    test_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    observed_value = 5.5

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )

    # Test boundary percentiles
    min_val = result.percentile(0)
    max_val = result.percentile(100)
    assert min_val == 1.0
    assert max_val == 10.0

    # Test median
    median = result.percentile(50)
    assert median == 5.5

    # Test quartiles
    q1 = result.percentile(25)
    q3 = result.percentile(75)
    assert q1 < median < q3

    # Test multiple percentiles at once
    percentiles = result.percentile([10, 25, 50, 75, 90])
    assert isinstance(percentiles, np.ndarray)
    assert len(percentiles) == 5
    assert all(
        percentiles[i] <= percentiles[i + 1] for i in range(len(percentiles) - 1)
    )

    # Test different interpolation methods
    methods = ["linear", "lower", "higher", "midpoint", "nearest"]
    for method in methods:
        p50 = result.percentile(50, method=method)
        assert np.isscalar(p50)

    # Test edge case with single percentile
    single_p = result.percentile(42.5)
    assert np.isscalar(single_p)


def test_percentile_edge_cases():
    """Test percentile method with edge cases."""
    # Test with small dataset
    test_values = [1.0, 2.0]
    observed_value = 1.5

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="bootstrap"
    )

    # Test with small dataset
    p50 = result.percentile(50)
    assert np.isscalar(p50)

    # Test with identical values
    test_values_identical = [5.0, 5.0, 5.0, 5.0, 5.0]
    result_identical = StatisticalTestingMixin._statistical_test_result(
        5.0,
        test_values_identical,
        n_tests=len(test_values_identical),
        method="permutation_test",
    )

    p25 = result_identical.percentile(25)
    p75 = result_identical.percentile(75)
    assert p25 == p75 == 5.0

    # Test with array input
    percentiles_array = result.percentile(np.array([25, 50, 75]))
    assert isinstance(percentiles_array, np.ndarray)
    assert len(percentiles_array) == 3


def test_confidence_interval_basic():
    """Test basic confidence interval functionality."""
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    observed_value = 0.55

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )

    # Test 95% confidence interval
    ci_95 = result.confidence_interval(95)
    assert isinstance(ci_95, np.ndarray)
    assert len(ci_95) == 2
    assert ci_95[0] <= ci_95[1]  # Lower bound <= upper bound

    # Verify it matches manual percentile calculation
    manual_ci_95 = result.percentile([2.5, 97.5])
    assert np.array_equal(ci_95, manual_ci_95)

    # Test 90% confidence interval
    ci_90 = result.confidence_interval(90)
    assert isinstance(ci_90, np.ndarray)
    assert len(ci_90) == 2
    assert ci_90[0] <= ci_90[1]

    # Verify it matches manual percentile calculation
    manual_ci_90 = result.percentile([5, 95])
    assert np.array_equal(ci_90, manual_ci_90)

    # Test 99% confidence interval
    ci_99 = result.confidence_interval(99)
    assert isinstance(ci_99, np.ndarray)
    assert len(ci_99) == 2
    assert ci_99[0] <= ci_99[1]

    # Verify it matches manual percentile calculation
    manual_ci_99 = result.percentile([0.5, 99.5])
    assert np.array_equal(ci_99, manual_ci_99)


def test_confidence_interval_ordering():
    """Test that wider confidence intervals are indeed wider."""
    test_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    observed_value = 5.5

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="bootstrap"
    )

    # Get different confidence intervals
    ci_80 = result.confidence_interval(80)
    ci_90 = result.confidence_interval(90)
    ci_95 = result.confidence_interval(95)
    ci_99 = result.confidence_interval(99)

    # Check that wider CIs are indeed wider
    width_80 = ci_80[1] - ci_80[0]
    width_90 = ci_90[1] - ci_90[0]
    width_95 = ci_95[1] - ci_95[0]
    width_99 = ci_99[1] - ci_99[0]

    assert width_99 >= width_95 >= width_90 >= width_80


def test_confidence_interval_methods():
    """Test confidence interval with different interpolation methods."""
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    observed_value = 0.3

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )

    # Test different interpolation methods
    methods = ["linear", "lower", "higher", "midpoint", "nearest"]
    for method in methods:
        ci = result.confidence_interval(95, method=method)
        assert isinstance(ci, np.ndarray)
        assert len(ci) == 2
        assert ci[0] <= ci[1]

        # Verify it matches manual percentile calculation
        manual_ci = result.percentile([2.5, 97.5], method=method)
        assert np.array_equal(ci, manual_ci)


def test_confidence_interval_edge_cases():
    """Test confidence interval with edge cases."""
    # Test with small dataset
    test_values = [1.0, 2.0]
    observed_value = 1.5

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="bootstrap"
    )

    # Test with small dataset
    ci_95 = result.confidence_interval(95)
    assert isinstance(ci_95, np.ndarray)
    assert len(ci_95) == 2

    # Test with identical values
    test_values_identical = [5.0, 5.0, 5.0, 5.0, 5.0]
    result_identical = StatisticalTestingMixin._statistical_test_result(
        5.0,
        test_values_identical,
        n_tests=len(test_values_identical),
        method="permutation_test",
    )

    ci_identical = result_identical.confidence_interval(95)
    assert isinstance(ci_identical, np.ndarray)
    assert len(ci_identical) == 2
    assert ci_identical[0] == ci_identical[1] == 5.0  # All values are the same

    # Test very narrow confidence interval
    ci_1 = result.confidence_interval(1)
    assert isinstance(ci_1, np.ndarray)
    assert len(ci_1) == 2
    assert ci_1[0] <= ci_1[1]

    # Test very wide confidence interval
    ci_99_9 = result.confidence_interval(99.9)
    assert isinstance(ci_99_9, np.ndarray)
    assert len(ci_99_9) == 2
    assert ci_99_9[0] <= ci_99_9[1]


def test_confidence_interval_invalid_levels():
    """Test error handling for invalid confidence levels."""
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    observed_value = 0.3

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )

    # Test confidence level >= 100
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 100"):
        result.confidence_interval(100)

    with pytest.raises(ValueError, match="Confidence level must be between 0 and 100"):
        result.confidence_interval(101)

    # Test confidence level <= 0
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 100"):
        result.confidence_interval(0)

    with pytest.raises(ValueError, match="Confidence level must be between 0 and 100"):
        result.confidence_interval(-5)


def test_confidence_interval_comprehensive():
    """Test comprehensive confidence interval functionality."""
    test_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    observed_value = 5.5

    result = StatisticalTestingMixin._statistical_test_result(
        observed_value, test_values, n_tests=len(test_values), method="permutation_test"
    )

    # Test common confidence levels
    common_levels = [50, 68, 80, 90, 95, 99, 99.9]
    for level in common_levels:
        ci = result.confidence_interval(level)
        assert isinstance(ci, np.ndarray)
        assert len(ci) == 2
        assert ci[0] <= ci[1]

        # Verify calculation is correct
        y = (100 - level) / 2
        expected_ci = result.percentile([y, 100 - y])
        assert np.array_equal(ci, expected_ci)

    # Test fractional confidence levels
    fractional_levels = [95.5, 99.95, 68.27]
    for level in fractional_levels:
        ci = result.confidence_interval(level)
        assert isinstance(ci, np.ndarray)
        assert len(ci) == 2
        assert ci[0] <= ci[1]
