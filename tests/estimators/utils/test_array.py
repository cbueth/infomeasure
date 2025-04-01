"""Tests for the array helpers."""

import pytest
from numpy import array, array_equal
from infomeasure.estimators.utils.array import assure_2d_data


@pytest.mark.parametrize(
    "data, expected",
    [
        # list
        ([1, 2, 3], array([[1], [2], [3]])),  # convert 1D to 2D
        (([[1], [2], [3]]), array([[1], [2], [3]])),  # keep 2D as is
        (([[1, 2], [3, 4]]), array([[1, 2], [3, 4]])),  # keep 2D as is
        # array-like
        (array([1, 2, 3]), array([[1], [2], [3]])),  # convert 1D to 2D
        (array([[1, 2], [3, 4]]), array([[1, 2], [3, 4]])),  # keep 2D as is
        # Tuple
        (  # convert 1D to 2D
            (array([1, 2, 3]), array([4, 5, 6])),
            (array([[1], [2], [3]]), array([[4], [5], [6]])),
        ),
        (  # keep 2D as is
            (array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])),
            (array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])),
        ),
        # Generator
        (  # convert 1D to 2D
            (x for x in (array([1, 2, 3]), array([4, 5, 6]))),
            (array([[1], [2], [3]]), array([[4], [5], [6]])),
        ),
        (  # keep 2D as is
            (x for x in (array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]]))),
            (array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])),
        ),
    ],
)
def test_assure_2d_data(data, expected):
    result = assure_2d_data(data)
    assert array_equal(result, expected)


def test_assure_2d_data_with_unsupported_type():
    data = "unsupported"
    with pytest.raises(ValueError):
        assure_2d_data(data)
