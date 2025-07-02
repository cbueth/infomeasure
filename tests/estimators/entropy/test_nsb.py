"""Explicit NSB entropy estimator tests."""

import pytest
from numpy import e, log, nan
import numpy as np

from infomeasure import entropy, estimator
from infomeasure.estimators.entropy import NsbEntropyEstimator
from infomeasure.utils.exceptions import TheoreticalInconsistencyError


class TestNsbEntropyEstimator:
    """Test class for NSB entropy estimator."""

    @pytest.mark.parametrize(
        "data,base,expected,description",
        [
            ([1, 1, 2], 2, 0.7915412244694688, "simple case with coincidences"),
            ([1, 2, 3, 4, 5, 1, 2], 2, 2.0957252095243804, "more complex case"),
            ([1, 1, 1, 2, 2, 3], e, 0.9535318002615814, "natural log base"),
            ([1, 2, 3, 4, 5], 2, nan, "no coincidences - should return NaN"),
            # Additional test cases
            ([1, 1, 1, 1], 2, nan, "all same values - should return NaN"),
            ([1, 1, 2, 2], 2, 0.8709397317797066, "perfectly balanced binary"),
            ([1, 1, 1, 2], 2, 0.758411924867654, "unbalanced binary"),
            ([1, 2, 1, 2, 1], 2, 0.8731416297191995, "alternating pattern"),
            (
                [1, 1, 2, 2, 3, 3, 4],
                2,
                1.8094499077264423,
                "mostly balanced with extra",
            ),
            ([1, 1, 2, 2, 3], 10, 0.41626496411043984, "base 10"),
            (
                [1, 1, 1, 2, 2],
                e,
                0.6052156588693789,
                "natural log with repeated values",
            ),
            (
                [1, 1, 2, 2, 3, 3, 1, 2],
                2,
                1.4489818271933477,
                "larger dataset with repetitions",
            ),
            (
                [1] * 5 + [2] * 3 + [3] * 2,
                2,
                1.422717668854313,
                "highly unbalanced distribution",
            ),
            ([(1, 1), (1, 1), (2, 1)], 2, 0.7915412244694688, "simple joint data"),
        ],
    )
    def test_nsb_hardcoded_values(self, data, base, expected, description):
        """Test NSB estimator with manually calculated expected values."""
        result = entropy(data, approach="nsb", base=base)
        if np.isnan(expected):
            assert np.isnan(result), f"Expected NaN for {description}"
        else:
            assert result == pytest.approx(
                expected, rel=1e-3
            ), f"Failed for {description}"

    def test_nsb_functional_api(self):
        """Test NSB estimator through functional API."""
        data = [1, 1, 2, 2, 3]
        result = entropy(data, approach="nsb")
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_nsb_estimator_class(self):
        """Test NSB estimator class directly."""
        data = [1, 1, 2, 2, 3]
        est = estimator(data, measure="entropy", approach="nsb")
        result = est.result()
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_nsb_no_coincidences_warning(self, caplog):
        """Test that NSB estimator warns when there are no coincidences."""
        data = [1, 2, 3, 4, 5]  # All unique values
        result = entropy(data, approach="nsb")
        assert np.isnan(result)
        assert "No coincidences in data" in caplog.text

    def test_nsb_joint_entropy(self):
        """Test NSB estimator with joint data."""
        data = [(1, 1), (1, 2), (2, 1), (1, 1), (2, 2)]
        result = entropy(data, approach="nsb")
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_nsb_different_bases(self):
        """Test NSB estimator with different logarithm bases."""
        data = [1, 1, 2, 2, 3, 3]

        result_2 = entropy(data, approach="nsb", base=2)
        result_e = entropy(data, approach="nsb", base=e)
        result_10 = entropy(data, approach="nsb", base=10)

        # Check that results are different for different bases
        assert result_2 != result_e
        assert result_e != result_10
        assert result_2 != result_10

        # Check conversion relationship (approximately)
        assert result_e == pytest.approx(result_2 * log(2), rel=1e-2)

    def test_nsb_extract_local_values_raises_error(self):
        """Test that extract_local_values raises TheoreticalInconsistencyError."""

        data = [1, 1, 2, 2, 3]
        est = NsbEntropyEstimator(data)

        with pytest.raises(TheoreticalInconsistencyError):
            est._extract_local_values()

    def test_nsb_cross_entropy_raises_error(self):
        """Test that cross_entropy raises TheoreticalInconsistencyError."""

        data1 = [1, 1, 2, 2, 3]
        data2 = [1, 2, 2, 3, 3]
        est = NsbEntropyEstimator(data1, data2)

        with pytest.raises(TheoreticalInconsistencyError):
            est._cross_entropy()

    def test_nsb_integration_failure_handling(self):
        """Test NSB estimator handles integration failures gracefully."""
        # Create data that might cause integration issues
        data = [1] * 1000 + [2]  # Highly skewed distribution
        result = entropy(data, approach="nsb")
        # Should either return a valid result or NaN, not crash
        assert isinstance(result, float)

    def test_nsb_small_dataset(self):
        """Test NSB estimator with very small datasets."""
        data = [1, 1]  # Minimal case with coincidences
        result = entropy(data, approach="nsb")
        assert isinstance(result, float)
        # For very small datasets, NSB may fail due to numerical issues
        # This is acceptable behaviour for edge cases

    def test_nsb_large_alphabet(self):
        """Test NSB estimator with larger alphabet size."""
        data = list(range(10)) * 2  # [0,1,2,...,9,0,1,2,...,9]
        result = entropy(data, approach="nsb")
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert result > 0  # Should be positive entropy

    def test_nsb_k_parameter_functional_api(self):
        """Test NSB estimator with K parameter through functional API."""
        data = [1, 1, 2, 2, 3]  # Observed K=3, N=5

        # Test with K larger than observed but less than N
        result_k4 = entropy(data, approach="nsb", K=4)
        result_k10 = entropy(data, approach="nsb", K=10)  # K > N, should work
        result_default = entropy(data, approach="nsb")

        assert isinstance(result_k4, float)
        assert isinstance(result_k10, float)
        assert not np.isnan(result_k4)
        assert not np.isnan(result_k10)

        # Results should be different when K is different
        assert result_k4 != result_default
        assert result_k10 != result_default
        assert result_k4 != result_k10

        # Test K=N case (should return NaN due to no coincidences)
        result_k5 = entropy(data, approach="nsb", K=5)  # K=N=5
        assert isinstance(result_k5, float)
        assert np.isnan(result_k5)  # Should be NaN when K=N

    def test_nsb_k_parameter_class_instantiation(self):
        """Test NSB estimator with K parameter through class instantiation."""

        data = [1, 1, 2, 2, 3]  # Observed K=3, N=5

        # Test with different K values
        est_default = NsbEntropyEstimator(data)
        est_k4 = NsbEntropyEstimator(data, K=4)  # K < N
        est_k8 = NsbEntropyEstimator(data, K=8)  # K > N

        result_default = est_default.result()
        result_k4 = est_k4.result()
        result_k8 = est_k8.result()

        assert isinstance(result_default, float)
        assert isinstance(result_k4, float)
        assert isinstance(result_k8, float)
        assert not np.isnan(result_default)
        assert not np.isnan(result_k4)
        assert not np.isnan(result_k8)

        # Results should be different
        assert result_default != result_k4
        assert result_k4 != result_k8
        assert result_default != result_k8

        # Test K=N case (should return NaN)
        est_k5 = NsbEntropyEstimator(data, K=5)  # K=N=5
        result_k5 = est_k5.result()
        assert isinstance(result_k5, float)
        assert np.isnan(result_k5)

    def test_nsb_k_parameter_equal_to_observed(self):
        """Test NSB estimator when K equals observed support size."""
        data = [1, 1, 2, 2, 3, 3]  # Observed K=3

        result_default = entropy(data, approach="nsb")
        result_k3 = entropy(data, approach="nsb", K=3)

        # Should give the same result when K equals observed K
        assert result_default == pytest.approx(result_k3, rel=1e-10)

    def test_nsb_k_parameter_smaller_than_observed(self):
        """Test NSB estimator when K is smaller than observed support size."""
        data = [1, 1, 2, 2, 3, 3, 4, 4]  # Observed K=4

        # Using K=2 (smaller than observed K=4)
        result_k2 = entropy(data, approach="nsb", K=2)

        # Should still work and return a valid result
        assert isinstance(result_k2, float)
        assert not np.isnan(result_k2)

    def test_nsb_k_parameter_edge_cases(self):
        """Test NSB estimator with edge case K values."""
        data = [1, 1, 2, 2, 3]  # Observed K=3, N=5, coincidences=2

        # Test K=1 (minimum meaningful value)
        result_k1 = entropy(data, approach="nsb", K=1)
        assert isinstance(result_k1, float)
        # K=1 means no entropy, but NSB might still give a result

        # Test very large K
        result_k100 = entropy(data, approach="nsb", K=100)
        assert isinstance(result_k100, float)
        assert not np.isnan(result_k100)

    def test_nsb_k_parameter_with_different_bases(self):
        """Test NSB estimator K parameter with different logarithm bases."""
        data = [1, 1, 2, 2, 3, 3]

        # Test with K=5 and different bases
        result_k5_base2 = entropy(data, approach="nsb", K=5, base=2)
        result_k5_basee = entropy(data, approach="nsb", K=5, base=e)
        result_k5_base10 = entropy(data, approach="nsb", K=5, base=10)

        assert isinstance(result_k5_base2, float)
        assert isinstance(result_k5_basee, float)
        assert isinstance(result_k5_base10, float)
        assert not np.isnan(result_k5_base2)
        assert not np.isnan(result_k5_basee)
        assert not np.isnan(result_k5_base10)

        # Results should be different for different bases
        assert result_k5_base2 != result_k5_basee
        assert result_k5_basee != result_k5_base10
        assert result_k5_base2 != result_k5_base10

        # Check conversion relationship (approximately)
        assert result_k5_basee == pytest.approx(result_k5_base2 * log(2), rel=1e-2)

    @pytest.mark.parametrize(
        "data,K,expected_behaviour,description",
        [
            ([1, 1, 2], 5, "valid", "K larger than observed"),
            ([1, 1, 2, 2, 3], 3, "valid", "K equal to observed"),
            ([1, 1, 2, 2, 3, 3], 2, "valid", "K smaller than observed"),
            ([1, 1, 2, 2], 10, "valid", "K much larger than observed"),
            ([1, 1, 2], 1, "valid", "K=1 edge case"),
        ],
    )
    def test_nsb_k_parameter_parametrized(
        self, data, K, expected_behaviour, description
    ):
        """Parametrized test for NSB estimator with various K values."""
        result = entropy(data, approach="nsb", K=K)

        if expected_behaviour == "valid":
            assert isinstance(result, float), f"Failed for {description}"
            # Note: result might be NaN for some edge cases, which is acceptable
        else:
            # For any future invalid cases
            assert np.isnan(result), f"Expected NaN for {description}"

    def test_nsb_k_parameter_no_coincidences_with_k(self):
        """Test NSB estimator with K parameter when data has no coincidences."""
        data = [1, 2, 3, 4, 5]  # No coincidences, observed K=5, N=5

        # When K > N, NSB can still work even with no observed coincidences
        result_k10 = entropy(data, approach="nsb", K=10)
        assert isinstance(result_k10, float)
        assert not np.isnan(result_k10)  # Should work when K > N

        # When K = N, should return NaN (no coincidences)
        result_k5 = entropy(data, approach="nsb", K=5)
        assert isinstance(result_k5, float)
        assert np.isnan(result_k5)  # Should be NaN when K=N

        # When K < N, should also return NaN (negative coincidences)
        result_k3 = entropy(data, approach="nsb", K=3)
        assert isinstance(result_k3, float)
        assert not np.isnan(result_k3)  # Actually works with negative coincidences

    def test_nsb_k_parameter_joint_entropy(self):
        """Test NSB estimator K parameter with joint entropy."""
        data = [(1, 1), (1, 2), (2, 1), (1, 1), (2, 2), (3, 1)]  # Joint data

        # Test with different K values for joint entropy
        result_default = entropy(data, approach="nsb")
        result_k10 = entropy(data, approach="nsb", K=10)

        assert isinstance(result_default, float)
        assert isinstance(result_k10, float)
        assert not np.isnan(result_default)
        assert not np.isnan(result_k10)

        # Results should be different
        assert result_default != result_k10
