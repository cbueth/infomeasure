"""Test the utilities modules for unique values."""

import pytest
from numpy import array
from numpy.testing import assert_array_almost_equal

from infomeasure.estimators.utils.unique import histogram_unique_values


@pytest.mark.parametrize(
    "data, expected_histogram, expected_dist_dict",
    [
        (
            [1, 2, 2, 3, 3, 3],
            [1 / 6, 2 / 6, 3 / 6],
            {1: 1 / 6, 2: 2 / 6, 3: 3 / 6},
        ),
        ([1, 1, 1, 1], [1.0], {1: 1.0}),
        (
            [1, 2, 3, 4, 5],
            [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5],
            {1: 1 / 5, 2: 1 / 5, 3: 1 / 5, 4: 1 / 5, 5: 1 / 5},
        ),
        (
            [1, 1, 2, 2, 2, 3],
            [2 / 6, 3 / 6, 1 / 6],
            {1: 2 / 6, 2: 3 / 6, 3: 1 / 6},
        ),
        (
            list(range(0, 40)),
            [1 / 40] * 40,
            {i: 1 / 40 for i in range(0, 40)},
        ),
        (
            ["a", "b", "b", "c", "c", "c"],
            [1 / 6, 2 / 6, 3 / 6],
            {"a": 1 / 6, "b": 2 / 6, "c": 3 / 6},
        ),
        (
            [1.0, 4.2, 4.2, 3.0, 3.0, 3.0],
            [1 / 6, 3 / 6, 2 / 6],
            {1.0: 1 / 6, 4.2: 2 / 6, 3.0: 3 / 6},
        ),
        ([1], [1.0], {1: 1.0}),
    ],
)
def test_histogram_unique_values(data, expected_histogram, expected_dist_dict):
    """Test the histogram of unique values by design."""
    histogram, dist_dict = histogram_unique_values(array(data))
    assert_array_almost_equal(histogram, array(expected_histogram))
    assert dist_dict == expected_dist_dict


def test_histogram_unique_values_empty():
    """Test the histogram of unique values for an empty array."""
    histogram, dist_dict = histogram_unique_values(array([]))
    assert_array_almost_equal(histogram, array([]))
    assert dist_dict == {}
