"""Tests for discrete transfer entropy utility functions."""

import pytest
from numpy import array, log, log2, log10, ndarray, mean
from numpy.random import default_rng
from numpy.testing import assert_allclose

from infomeasure.estimators.utils.discrete_transfer_entropy import combined_te_form
from infomeasure.estimators.utils.te_slicing import te_observations, cte_observations


# Mock slice method for testing
def mock_slice_method_3_outputs(*data, **kwargs):
    """Mock slice method that returns 3 arrays (for TE without conditional)."""
    # Simple mock that returns the first 3 data arrays as-is
    return data[0][:10], data[1][:10], data[0][1:11]


def mock_slice_method_4_outputs(*data, **kwargs):
    """Mock slice method that returns 4 arrays (for CTE with conditional)."""
    # Simple mock that returns 4 arrays
    return data[0][:10], data[1][:10], data[0][1:11], data[2][:10]


def mock_slice_method_invalid(*data, **kwargs):
    """Mock slice method that returns invalid number of arrays."""
    return data[0][:10], data[1][:10]  # Only 2 arrays - should cause error


class TestCombinedTeForm:
    """Test the combined_te_form function."""

    def test_basic_functionality_with_te_observations(self):
        """Test basic functionality with te_observations slice method."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 4, 100)
        dest = rng.integers(0, 4, 100)

        # Act: Calculate transfer entropy
        result = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            src_hist_len=2,
            dest_hist_len=2,
            step_size=1,
        )

        # Assert: Result should be a float
        assert isinstance(result, float)
        assert not isinstance(result, ndarray)

    def test_basic_functionality_with_cte_observations(self):
        """Test basic functionality with cte_observations slice method."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 4, 100)
        dest = rng.integers(0, 4, 100)
        cond = rng.integers(0, 4, 100)

        # Act: Calculate conditional transfer entropy
        result = combined_te_form(
            cte_observations,
            source,
            dest,
            cond,
            local=False,
            src_hist_len=2,
            dest_hist_len=2,
            cond_hist_len=2,
            step_size=1,
        )

        # Assert: Result should be a float
        assert isinstance(result, float)
        assert not isinstance(result, ndarray)

    def test_local_vs_global_calculation(self):
        """Test that local and global calculations return different types."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 4, 50)
        dest = rng.integers(0, 4, 50)

        # Act: Calculate both local and global
        global_result = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )
        local_result = combined_te_form(
            te_observations,
            source,
            dest,
            local=True,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Global should be float, local should be array
        assert isinstance(global_result, float)
        assert isinstance(local_result, ndarray)
        assert local_result.ndim == 1
        assert len(local_result) > 0
        assert mean(local_result) == pytest.approx(global_result)

    @pytest.mark.parametrize(
        "log_func, base",
        [
            (log, "e"),
            (log2, 2),
            (log10, 10),
        ],
    )
    def test_different_log_functions(self, log_func, base):
        """Test with different logarithm functions."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 3, 50)
        dest = rng.integers(0, 3, 50)

        # Act: Calculate transfer entropy with different log functions
        result = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            log_func=log_func,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Result should be a finite float
        assert isinstance(result, float)
        assert not (result != result)  # Check for NaN
        assert abs(result) != float("inf")  # Check for infinity

    def test_miller_madow_correction_applied(self):
        """Test that Miller-Madow correction changes the result."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 3, 100)
        dest = rng.integers(0, 3, 100)

        # Act: Calculate with and without Miller-Madow correction
        result_no_correction = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            src_hist_len=2,
            dest_hist_len=2,
            step_size=1,
        )
        result_with_correction = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            miller_madow_correction=2,
            src_hist_len=2,
            dest_hist_len=2,
            step_size=1,
        )

        # Assert: Results should be different
        assert result_no_correction != result_with_correction
        assert isinstance(result_with_correction, float)

    @pytest.mark.parametrize(
        "correction_value",
        [2, 10, "e", 2.0, 10.0],
    )
    def test_miller_madow_correction_different_bases(self, correction_value):
        """Test Miller-Madow correction with different base values."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 3, 50)
        dest = rng.integers(0, 3, 50)

        # Act: Calculate with Miller-Madow correction
        result = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            miller_madow_correction=correction_value,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Result should be a finite float
        assert isinstance(result, float)
        assert not (result != result)  # Check for NaN
        assert abs(result) != float("inf")  # Check for infinity

    def test_mock_slice_method_3_outputs(self):
        """Test with mock slice method returning 3 outputs."""
        # Arrange: Create simple test data
        source = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        dest = array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        # Act: Use mock slice method
        result = combined_te_form(
            mock_slice_method_3_outputs,
            source,
            dest,
            local=False,
        )

        # Assert: Should return a float result
        assert isinstance(result, float)

    def test_mock_slice_method_4_outputs(self):
        """Test with mock slice method returning 4 outputs."""
        # Arrange: Create simple test data
        source = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        dest = array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        cond = array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        # Act: Use mock slice method
        result = combined_te_form(
            mock_slice_method_4_outputs,
            source,
            dest,
            cond,
            local=False,
        )

        # Assert: Should return a float result
        assert isinstance(result, float)

    def test_invalid_slice_method_raises_error(self):
        """Test that invalid slice method output raises ValueError."""
        # Arrange: Create simple test data
        source = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        dest = array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError, match="Invalid number of data arrays"):
            combined_te_form(
                mock_slice_method_invalid,
                source,
                dest,
                local=False,
            )

    def test_consistent_results_with_same_seed(self):
        """Test that results are consistent with the same random seed."""
        # Arrange: Create test data with same seed
        rng1 = default_rng(123)
        source1 = rng1.integers(0, 4, 50)
        dest1 = rng1.integers(0, 4, 50)

        rng2 = default_rng(123)
        source2 = rng2.integers(0, 4, 50)
        dest2 = rng2.integers(0, 4, 50)

        # Act: Calculate transfer entropy with same parameters
        result1 = combined_te_form(
            te_observations,
            source1,
            dest1,
            local=False,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )
        result2 = combined_te_form(
            te_observations,
            source2,
            dest2,
            local=False,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Results should be identical
        assert result1 == result2

    def test_local_calculation_with_miller_madow(self):
        """Test local calculation with Miller-Madow correction."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 3, 50)
        dest = rng.integers(0, 3, 50)

        # Act: Calculate local values with Miller-Madow correction
        result = combined_te_form(
            te_observations,
            source,
            dest,
            local=True,
            miller_madow_correction=2,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Should return array of local values
        assert isinstance(result, ndarray)
        assert result.ndim == 1
        assert len(result) > 0
        # All values should be finite
        assert all(not (val != val) for val in result)  # Check for NaN
        assert all(abs(val) != float("inf") for val in result)  # Check for infinity

    def test_conditional_transfer_entropy_with_miller_madow(self):
        """Test conditional transfer entropy with Miller-Madow correction."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 3, 100)
        dest = rng.integers(0, 3, 100)
        cond = rng.integers(0, 3, 100)

        # Act: Calculate CTE with Miller-Madow correction
        result = combined_te_form(
            cte_observations,
            source,
            dest,
            cond,
            local=False,
            miller_madow_correction="e",
            src_hist_len=1,
            dest_hist_len=1,
            cond_hist_len=1,
            step_size=1,
        )

        # Assert: Should return a finite float
        assert isinstance(result, float)
        assert not (result != result)  # Check for NaN
        assert abs(result) != float("inf")  # Check for infinity

    def test_slice_kwargs_passed_correctly(self):
        """Test that slice_kwargs are passed correctly to slice method."""
        # Arrange: Create simple test data
        rng = default_rng(42)
        source = rng.integers(0, 4, 100)
        dest = rng.integers(0, 4, 100)

        # Act: Calculate with specific slice parameters
        result = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            src_hist_len=3,
            dest_hist_len=2,
            step_size=2,
        )

        # Assert: Should complete without error and return float
        assert isinstance(result, float)

    def test_edge_case_minimal_data(self):
        """Test with minimal data that still allows calculation."""
        # Arrange: Create minimal test data
        source = array([0, 1, 0, 1, 0, 1])
        dest = array([1, 0, 1, 0, 1, 0])

        # Act: Calculate with minimal parameters
        result = combined_te_form(
            te_observations,
            source,
            dest,
            local=False,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Should return a finite float
        assert isinstance(result, float)
        assert not (result != result)  # Check for NaN
        assert abs(result) != float("inf")  # Check for infinity

    def test_different_data_types(self):
        """Test with different data types (int, float, string)."""
        # Test with integer data
        source_int = array([1, 2, 3, 1, 2, 3, 1, 2])
        dest_int = array([3, 1, 2, 3, 1, 2, 3, 1])

        result_int = combined_te_form(
            te_observations,
            source_int,
            dest_int,
            local=False,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Test with string data
        source_str = array(["a", "b", "c", "a", "b", "c", "a", "b"])
        dest_str = array(["c", "a", "b", "c", "a", "b", "c", "a"])

        result_str = combined_te_form(
            te_observations,
            source_str,
            dest_str,
            local=False,
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
        )

        # Assert: Both should return finite floats
        assert isinstance(result_int, float)
        assert isinstance(result_str, float)
        assert not (result_int != result_int)  # Check for NaN
        assert not (result_str != result_str)  # Check for NaN
