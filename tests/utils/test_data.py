"""Comprehensive tests for the DiscreteData dataclass."""

import pytest
import numpy as np

from infomeasure.utils.data import DiscreteData


class TestDiscreteData:
    """Test suite for DiscreteData dataclass."""

    def test_from_data_basic(self):
        """Test basic functionality of from_data class method."""
        data = np.array([1, 2, 2, 3, 3, 3])
        discrete_data = DiscreteData.from_data(data)

        # Check unique values and counts
        expected_uniq = np.array([1, 2, 3])
        expected_counts = np.array([1, 2, 3])

        np.testing.assert_array_equal(discrete_data.uniq, expected_uniq)
        np.testing.assert_array_equal(discrete_data.counts, expected_counts)
        np.testing.assert_array_equal(discrete_data.data, data)

        # Check computed fields
        assert discrete_data.N == 6
        assert discrete_data.K == 3

    def test_from_data_single_value(self):
        """Test from_data with single unique value."""
        data = np.array([5, 5, 5, 5])
        discrete_data = DiscreteData.from_data(data)

        np.testing.assert_array_equal(discrete_data.uniq, np.array([5]))
        np.testing.assert_array_equal(discrete_data.counts, np.array([4]))
        assert discrete_data.N == 4
        assert discrete_data.K == 1

    def test_from_data_empty_array(self):
        """Test from_data with empty array raises ValueError."""
        data = np.array([])
        with pytest.raises(ValueError, match="`data` must not be empty"):
            DiscreteData.from_data(data)

    def test_from_counts_basic(self):
        """Test basic functionality of from_counts class method."""
        uniq = np.array([1, 2, 3])
        counts = np.array([1, 2, 3])
        discrete_data = DiscreteData.from_counts(uniq, counts)

        np.testing.assert_array_equal(discrete_data.uniq, uniq)
        np.testing.assert_array_equal(discrete_data.counts, counts)
        assert discrete_data.data is None
        assert discrete_data.N == 6  # sum of counts
        assert discrete_data.K == 3  # len of uniq

    def test_from_counts_single_value(self):
        """Test from_counts with single unique value."""
        uniq = np.array([42])
        counts = np.array([10])
        discrete_data = DiscreteData.from_counts(uniq, counts)

        np.testing.assert_array_equal(discrete_data.uniq, uniq)
        np.testing.assert_array_equal(discrete_data.counts, counts)
        assert discrete_data.N == 10
        assert discrete_data.K == 1

    def test_validation_mismatched_lengths(self):
        """Test validation fails when uniq and counts have different lengths."""
        uniq = np.array([1, 2, 3])
        counts = np.array([1, 2])  # Different length

        with pytest.raises(ValueError, match="uniq and counts must have same length"):
            DiscreteData.from_counts(uniq, counts)

    def test_validation_negative_counts(self):
        """Test validation fails with negative counts."""
        uniq = np.array([1, 2, 3])
        counts = np.array([1, -1, 3])  # Negative count

        with pytest.raises(ValueError, match="counts must be non-negative"):
            DiscreteData.from_counts(uniq, counts)

    def test_validation_zero_sum_counts(self):
        """Test validation fails when counts sum to zero."""
        uniq = np.array([1, 2, 3])
        counts = np.array([0, 0, 0])  # Sum to zero

        with pytest.raises(ValueError, match="counts must sum to a positive value"):
            DiscreteData.from_counts(uniq, counts)

    def test_validation_data_counts_mismatch(self):
        """Test validation fails when data length doesn't match counts sum."""
        data = np.array([1, 2, 2, 3])  # Length 4
        uniq = np.array([1, 2, 3])
        counts = np.array([1, 2, 2])  # Sum to 5, not 4

        with pytest.raises(ValueError, match="counts must sum to the length of data"):
            DiscreteData(uniq=uniq, counts=counts, data=data)

    def test_validation_empty_data_direct_constructor(self):
        """Test validation fails when empty data is passed to direct constructor."""
        uniq = np.array([1, 2, 3])
        counts = np.array([1, 2, 3])
        empty_data = np.array([])  # Empty data array

        with pytest.raises(ValueError, match="data must not be empty"):
            DiscreteData(uniq=uniq, counts=counts, data=empty_data)

    def test_probabilities_property(self):
        """Test probabilities property calculation."""
        uniq = np.array([1, 2, 3])
        counts = np.array([2, 4, 6])
        discrete_data = DiscreteData.from_counts(uniq, counts)

        expected_probs = np.array([2 / 12, 4 / 12, 6 / 12])
        np.testing.assert_array_almost_equal(
            discrete_data.probabilities, expected_probs
        )

    def test_probabilities_property_single_value(self):
        """Test probabilities property with single value."""
        uniq = np.array([5])
        counts = np.array([10])
        discrete_data = DiscreteData.from_counts(uniq, counts)

        expected_probs = np.array([1.0])
        np.testing.assert_array_almost_equal(
            discrete_data.probabilities, expected_probs
        )

    def test_frozen_dataclass(self):
        """Test that DiscreteData is frozen (immutable)."""
        discrete_data = DiscreteData.from_counts(np.array([1, 2]), np.array([1, 1]))

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            discrete_data.uniq = np.array([3, 4])

        with pytest.raises(AttributeError):
            discrete_data.N = 10

    def test_k_field_consistency(self):
        """Test that K field is always consistent with len(uniq)."""
        # Test with various data sizes
        test_cases = [
            np.array([1]),  # Single value
            np.array([1, 2]),  # Two values
            np.array([1, 1, 2, 2, 3, 3, 3]),  # Multiple repeats
            np.array([5, 1, 3, 2, 4]),  # Unsorted unique values
        ]

        for data in test_cases:
            discrete_data = DiscreteData.from_data(data)
            assert discrete_data.K == len(discrete_data.uniq)
            assert discrete_data.K == len(np.unique(data))

    def test_k_field_from_counts(self):
        """Test K field when using from_counts method."""
        test_cases = [
            (np.array([1]), np.array([5])),  # Single unique value
            (np.array([1, 2, 3]), np.array([1, 2, 3])),  # Three unique values
            (
                np.array([10, 20, 30, 40, 50]),
                np.array([1, 1, 1, 1, 1]),
            ),  # Five unique values
        ]

        for uniq, counts in test_cases:
            discrete_data = DiscreteData.from_counts(uniq, counts)
            assert discrete_data.K == len(uniq)

    def test_string_data(self):
        """Test DiscreteData with string data."""
        data = np.array(["a", "b", "b", "c", "c", "c"])
        discrete_data = DiscreteData.from_data(data)

        expected_uniq = np.array(["a", "b", "c"])
        expected_counts = np.array([1, 2, 3])

        np.testing.assert_array_equal(discrete_data.uniq, expected_uniq)
        np.testing.assert_array_equal(discrete_data.counts, expected_counts)
        assert discrete_data.N == 6
        assert discrete_data.K == 3

    def test_float_data(self):
        """Test DiscreteData with float data."""
        data = np.array([1.0, 2.5, 2.5, 3.7, 3.7, 3.7])
        discrete_data = DiscreteData.from_data(data)

        expected_uniq = np.array([1.0, 2.5, 3.7])
        expected_counts = np.array([1, 2, 3])

        np.testing.assert_array_equal(discrete_data.uniq, expected_uniq)
        np.testing.assert_array_equal(discrete_data.counts, expected_counts)
        assert discrete_data.N == 6
        assert discrete_data.K == 3

    def test_large_dataset(self):
        """Test DiscreteData with larger dataset."""
        # Create data with 1000 samples and 10 unique values
        np.random.seed(42)
        data = np.random.randint(0, 10, 1000)
        discrete_data = DiscreteData.from_data(data)

        # Check that K is correct
        assert discrete_data.K == len(np.unique(data))
        assert discrete_data.N == 1000
        assert discrete_data.counts.sum() == 1000

        # Check probabilities sum to 1
        assert abs(discrete_data.probabilities.sum() - 1.0) < 1e-10

    @pytest.mark.parametrize(
        "uniq,counts,expected_k,expected_n",
        [
            (np.array([1]), np.array([5]), 1, 5),
            (np.array([1, 2]), np.array([3, 7]), 2, 10),
            (np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1]), 4, 4),
            (np.array([10, 20, 30]), np.array([2, 5, 3]), 3, 10),
        ],
    )
    def test_parametrized_k_and_n_values(self, uniq, counts, expected_k, expected_n):
        """Test K and N values with parametrized inputs."""
        discrete_data = DiscreteData.from_counts(uniq, counts)
        assert discrete_data.K == expected_k
        assert discrete_data.N == expected_n
