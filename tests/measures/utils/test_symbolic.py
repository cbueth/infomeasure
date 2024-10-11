"""Tests for the symbolic / permutation utility functions."""

from itertools import permutations

import pytest
from numpy import (
    array,
    ndarray,
    uint8,
    uint16,
    arange,
    uint32,
    uint64,
    apply_along_axis,
    iinfo,
)
from numpy.testing import assert_array_equal
from scipy.special import factorial

from infomeasure.measures.utils.symbolic import symbolize_series, permutation_to_integer


@pytest.mark.parametrize(
    "series, order, step_size, expected",
    [
        ([1], 1, 1, [[0]]),
        ([1, 2], 2, 1, [[0, 1]]),
        ([1, 2, 3], 2, 1, [[0, 1]] * 2),
        ([1, 2, 3, 4], 2, 1, [[0, 1]] * 3),
        ([1, 2, 1, 4], 2, 1, [[0, 1], [1, 0], [0, 1]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 2, 1, [[0, 1]] * 8),
        ([-4, 2, 8, 3], 3, 1, [[0, 1, 2], [0, 2, 1]]),
        ([-4, 2, 8, 1], 3, 1, [[0, 1, 2], [2, 0, 1]]),
        ([0, 0, -1, 1, -2, 2, -3, 3], 2, 2, [[1, 0], [0, 1]] * 3),
        ([0, 0, -1, 1, -2, 2, -3, 3], 2, 4, [[1, 0], [0, 1]] * 2),
        ([0, 0, -1, 1, -2, 2, -3, 3], 2, 6, [[1, 0], [0, 1]]),
        (
            [-1e-6, 0, -1, 1, -2, 2, -3, 3],
            6,
            1,
            [[4, 2, 0, 1, 3, 5], [5, 3, 1, 0, 2, 4], [4, 2, 0, 1, 3, 5]],
        ),
        ([3.2, 2.1, -1.2, -3.4, 0.0], 2, 1, [[1, 0]] * 3 + [[0, 1]]),
        (["a", "b", "c", "d"], 2, 1, [[0, 1]] * 3),
    ],
)
def test_symbolize_series(series, order, step_size, expected):
    """Test the symbolize_series function."""
    series = array(series)
    patterns = symbolize_series(series, order, step_size)
    assert_array_equal(patterns, array(expected))


@pytest.mark.parametrize(
    "series, order, expected",
    [
        ([1], 1, [0]),
        ([1, 2], 2, [0]),
        ([2, 1], 2, [1]),
        ([2, 1, 2], 2, [1, 0]),
        ([1, 2, 1], 2, [0, 1]),
        ([1, 2, 3, 4], 2, [0, 0, 0]),
        ([1, 2, 3], 3, [0]),
        ([1, 2, 3, 4], 3, [0, 0]),
        ([1, 2, 3, 4, 5, -3], 5, [0, 96]),
        ([1, 2, 3, 4, 5, 6], 6, [0]),
    ],
)
def test_symbolize_series_to_int(series, order, expected):
    """Test the symbolize_series function with to_int=True."""
    series = array(series)
    patterns = symbolize_series(series, order, to_int=True)
    assert_array_equal(patterns, array(expected))
    assert patterns.dtype == uint8 if order < 6 else uint16


@pytest.mark.parametrize("data_len", [10, int(1e3)])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("step_size", [1, 2, 3])
@pytest.mark.parametrize("to_int", [True, False])
def test_symbolize_series_programmatic(data_len, order, step_size, to_int, default_rng):
    """Test the symbolize_series function programmatically."""
    series = default_rng.normal(size=data_len)
    symbols = symbolize_series(series, order, step_size, to_int=to_int)
    assert isinstance(symbols, ndarray)
    # if to_int: False, check that shape is correct
    if not to_int:
        assert symbols.shape == (data_len - (order - 1) * step_size, order)
    # if to_int: True, check that shape is correct
    else:
        assert symbols.shape == (data_len - (order - 1) * step_size,)
        assert symbols.dtype == uint8


@pytest.mark.parametrize(
    "series, order, step_size, error",
    [
        ([], 2, 1, AttributeError),  # input is list, not ndarray
        (tuple([1]), 2, 1, AttributeError),  # input is tuple, not ndarray
        (array([1]), 1.0, 1, TypeError),  # order is float
        (array([1]), 1, 1.0, TypeError),  # step_size is float
        (array([1]), 0, 1, ValueError),  # order is < 1
        (array([1]), -4, 1, ValueError),  # order is < 1
        (array([1]), 1, 0, ValueError),  # step_size is < 1
        (array([1]), 1, -3, ValueError),  # step_size is < 1
        (array([]), 1, 1, ValueError),  # empty series
        (array([1, 2]), 3, 1, ValueError),  # series is too small for order
        (array([1, 2]), 2, 2, ValueError),  # series is too small for step_size+order
    ],
)
def test_symbolize_series_error(series, order, step_size, error):
    """Test the symbolize_series function with an error."""
    with pytest.raises(error):
        symbolize_series(series, order, step_size)


@pytest.mark.parametrize(
    "patt, expected, dtype",
    [
        ([0, 1], 0, uint8),
        ([1, 0], 1, uint8),
        ([0, 1, 2], 0, uint8),
        ([2, 1, 0], 5, uint8),
        ([1, 0, 2], 2, uint8),
        ([2, 0, 1], 4, uint8),
        ([0, 1], 0, uint16),
        ([1, 0], 1, uint16),
        ([0, 1, 2], 0, uint16),
        ([2, 1, 0], 5, uint16),
        ([1, 0, 2], 2, uint16),
        ([2, 0, 1], 4, uint16),
    ],
)
def test_permutation_to_integer(patt, expected, dtype):
    """Test the permutation_to_integer function."""
    result = permutation_to_integer(array(patt), dtype=dtype)
    assert result == expected
    assert result.dtype == dtype


@pytest.mark.parametrize("patt_len", [1, 2, 5, 6, 8, 9, 12, 13, 20])
@pytest.mark.parametrize("dtype", [uint8, uint16, uint32, uint64])
def test_permutation_to_integer_min_max_patt(patt_len, dtype):
    """Test the permutation_to_integer function with min and max permutations."""
    patt = arange(patt_len)
    result = permutation_to_integer(patt, dtype=dtype)
    assert result == 0
    # Max value should be factorial(patt_len) - 1
    result_inv = permutation_to_integer(patt[::-1], dtype=dtype)
    if (
        # 5! < 2**8
        patt_len < 6  # 2**8 < 6! < 2**16
        or (patt_len < 9 and dtype == uint16)  # 2**16 < 9! < 2**32
        or (patt_len < 13 and dtype == uint32)  # 2**32 < 13! < 2**64
        or (patt_len < 21 and dtype == uint64)  # 2**64 < 21!
    ):  # No overflow
        assert result_inv == factorial(patt_len) - 1
    else:  # Overflow
        assert result_inv != factorial(patt_len) - 1

    assert result.dtype == dtype
    assert result_inv.dtype == dtype


def test_permutation_to_integer_order_too_large():
    """Test the permutation_to_integer function with an order that is too large."""
    with pytest.raises(ValueError):
        permutation_to_integer(arange(21))


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [uint8, uint16, uint32, uint64])
def test_permutation_to_integer_uniqueness(order, dtype):
    """Test the uniqueness property of the permutation_to_integer function."""
    # Get all possible permutations of [0, 1, 2, ..., order-1] without repetition
    perms = array(list(permutations(range(order))))
    results = apply_along_axis(permutation_to_integer, 1, perms, dtype=dtype)
    assert len(results) == factorial(order)
    assert results.dtype == dtype
    if (order < 6) or (order >= 6 and dtype != uint8):
        assert results.max() == factorial(order) - 1
        assert results.min() == 0
        # Check that all results are unique
        assert len(set(results)) == len(results)
    else:  # Overflow
        assert results.max() == iinfo(dtype).max
        assert results.min() == 0
        # The results cannot be unique,
        # as the number of permutations is larger than the dtype can handle
        assert len(set(results)) != len(results)
