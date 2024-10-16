"""Test the TE slicing utility functions."""

import pytest
from numpy import arange, array, hstack, array_equal
from numpy.random import default_rng
from numpy.testing import assert_equal

from infomeasure.measures.utils.te_slicing import (
    te_observations,
)


@pytest.mark.parametrize("data_len", [1, 2, 10, 100, 1e4])
@pytest.mark.parametrize("src_hist_len", [1, 2, 3])
@pytest.mark.parametrize("dest_hist_len", [1, 2, 3])
def test_te_observations_old_implementation(
    data_len, src_hist_len, dest_hist_len, step_size=1
):
    """Test the shape of the TE observations data arrays.

    Compare output to old/explicit implementation.
    The old implementation did not correctly implement the ``step_size`` subsampling.
    """
    src = arange(data_len)
    dest = arange(data_len, 2 * data_len)
    if max(src_hist_len, dest_hist_len) * step_size >= data_len:
        with pytest.raises(ValueError):
            te_observations(src, dest, src_hist_len, dest_hist_len, step_size)
        return
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(src, dest, src_hist_len, dest_hist_len, step_size)

    # Old implementation
    n = len(src)
    max_delay = max(src_hist_len * step_size, dest_hist_len * step_size)
    y_future = array([dest[i + max_delay] for i in range(n - max_delay)])
    y_history = array(
        [
            dest[i - dest_hist_len * step_size : i : step_size]
            for i in range(max_delay, n)
        ]
    )
    x_history = array(
        [src[i - src_hist_len * step_size : i : step_size] for i in range(max_delay, n)]
    )
    assert array_equal(
        joint_space_data, hstack((x_history, y_history, y_future.reshape(-1, 1)))
    )
    assert array_equal(data_dest_past_embedded, y_history)
    assert array_equal(marginal_1_space_data, hstack((x_history, y_history)))
    assert array_equal(
        marginal_2_space_data, hstack((y_history, y_future.reshape(-1, 1)))
    )


def test_te_observations():
    """Test the TE observations data arrays explicitly."""
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(
        arange(10), arange(10, 20), src_hist_len=3, dest_hist_len=2, step_size=2
    )
    assert (
        array(
            [
                [0, 2, 4, 12, 14, 16],
                [2, 4, 6, 14, 16, 18],
            ]
        )
        == joint_space_data
    ).all()
    assert (
        array(
            [
                [12, 14],
                [14, 16],
            ]
        )
        == data_dest_past_embedded
    ).all()
    assert (
        array(
            [
                [0, 2, 4, 12, 14],
                [2, 4, 6, 14, 16],
            ]
        )
        == marginal_1_space_data
    ).all()
    assert (
        array(
            [
                [12, 14, 16],
                [14, 16, 18],
            ]
        )
        == marginal_2_space_data
    ).all()


def test_te_observations_chars():
    """Test the TE observations data arrays with char arrays."""
    source = array(["a", "b", "c", "d"])
    destination = array(["e", "f", "g", "h"])
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(
        source, destination, src_hist_len=1, dest_hist_len=2, step_size=1
    )
    assert (
        array(
            [
                ["b", "e", "f", "g"],
                ["c", "f", "g", "h"],
            ]
        )
        == joint_space_data
    ).all()


def test_te_observations_tuple():
    """Test the TE observations data arrays with tuple arrays."""
    source = array([(1, 1), (2, 2), (3, 3), (4, 4)])
    destination = array([(5, 5), (6, 6), (7, 7), (8, 8)])
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(
        source, destination, src_hist_len=1, dest_hist_len=2, step_size=1
    )
    assert (
        array(
            [
                [(2, 2), (5, 5), (6, 6), (7, 7)],
                [(3, 3), (6, 6), (7, 7), (8, 8)],
            ]
        )
        == joint_space_data
    ).all()


@pytest.mark.parametrize(
    "data_len, src_hist_len, dest_hist_len, step_size",
    [
        (0, 1, 1, 1),
        (10, 0, 1, 1),
        (10, 1, 0, 1),
        (10, 1, 1, 0),
        (10, -1, 1, 1),
        (10, 1, -1, 1),
        (10, 1, 1, -1),
        (10, 1.0, 1, 1),
        (10, 1, 1.0, 1),
        (10, 1, 1, 1.0),
        (10, "1", 1, 1),
    ],
)
def test_te_observations_invalid_inputs(
    data_len, src_hist_len, dest_hist_len, step_size
):
    """Test the TE observations data arrays with invalid inputs."""
    with pytest.raises(ValueError):
        te_observations(
            arange(data_len), arange(data_len), src_hist_len, dest_hist_len, step_size
        )


@pytest.mark.parametrize(
    "src_hist_len, dest_hist_len, step_size",
    [(1, 1, 1), (2, 2, 2), (1, 4, 1), (4, 2, 1)],
)
@pytest.mark.parametrize("permute_src", [True, default_rng(5378)])
def test_te_observations_permute_src(
    src_hist_len, dest_hist_len, step_size, permute_src
):
    """Test the TE observations data arrays with permutation."""
    source = arange(10)
    destination = arange(10, 20)
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(
        source,
        destination,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        step_size=step_size,
        permute_src=False,
    )
    (
        joint_space_data_permuted,
        data_dest_past_embedded_permuted,
        marginal_1_space_data_permuted,
        marginal_2_space_data_permuted,
    ) = te_observations(
        source,
        destination,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        step_size=step_size,
        permute_src=permute_src,
    )
    assert array_equal(  # y_i^{(k)}, \hat{y}_{i+1} fixed
        joint_space_data[:, src_hist_len + 1 :],
        joint_space_data_permuted[:, src_hist_len + 1 :],
    )
    assert not array_equal(  # permuted x_i^{(l)}
        joint_space_data[:, :src_hist_len],
        joint_space_data_permuted[:, :src_hist_len],
    )
    assert array_equal(  # y_i^{(k)} fixed
        data_dest_past_embedded, data_dest_past_embedded_permuted
    )
    assert array_equal(  # y_i^{(k)} fixed
        marginal_1_space_data[:, src_hist_len:],
        marginal_1_space_data_permuted[:, src_hist_len:],
    )
    assert not array_equal(  # permuted x_i^{(l)}
        marginal_1_space_data[:, :src_hist_len],
        marginal_1_space_data_permuted[:, :src_hist_len],
    )
    assert array_equal(  # x_i^{(k)}, \hat{y}_{i+1} fixed
        marginal_2_space_data, marginal_2_space_data_permuted
    )


@pytest.mark.parametrize(
    "array_len,step_size,match_str",
    [
        (3, 1, "The history demanded"),
        (10, 1.0, "must be positive integers"),  # wrong type
        (10, 0, "must be positive integers"),  # non-positive
        (10, -1, "must be positive integers"),  # non-positive
    ],
)
def test_te_observations_value_errors(array_len, step_size, match_str):
    """Test the TE observations data arrays for value errors.

    - The demanded history is greater than the length of the data.
    - Both ``step_size_src`` or ``step_size_dest`` are set along with ``step_size``.
    - They are not positive integers.
    """
    with pytest.raises(ValueError, match=match_str):
        te_observations(
            arange(array_len),
            arange(array_len),
            src_hist_len=3,
            dest_hist_len=2,
            step_size=step_size,
        )


@pytest.mark.parametrize(  # old
    "src_hist_len, dest_hist_len, step_size, expected",
    [
        (1, 1, 1, array([0, 10, 11]) + arange(9)[:, None]),
        (1, 1, 2, array([0, 10, 12]) + arange(0, 8, 2)[:, None]),
        (1, 1, 3, array([0, 10, 13]) + arange(0, 7, 3)[:, None]),
        (2, 1, 1, array([0, 1, 11, 12]) + arange(8)[:, None]),
        (1, 2, 1, array([1, 10, 11, 12]) + arange(8)[:, None]),
        (1, 1, 2, array([0, 10, 12]) + arange(0, 8, 2)[:, None]),
        (2, 1, 2, array([0, 2, 12, 14]) + arange(0, 6, 2)[:, None]),
        (3, 1, 2, array([0, 2, 4, 14, 16]) + arange(0, 4, 2)[:, None]),
        (2, 2, 2, array([0, 2, 10, 12, 14]) + arange(0, 6, 2)[:, None]),
        (1, 2, 2, array([2, 10, 12, 14]) + arange(0, 6, 2)[:, None]),
        (3, 2, 2, array([0, 2, 4, 12, 14, 16]) + arange(0, 4, 2)[:, None]),
        (3, 2, 3, array([0, 3, 6, 13, 16, 19]) + arange(0, 1, 2)[:, None]),
    ],
)
def test_te_observations_step_sizes(src_hist_len, dest_hist_len, step_size, expected):
    """Test the TE observations data arrays with different step sizes."""
    joint_space_data, _, _, _ = te_observations(
        arange(10),
        arange(10, 20),
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        step_size=step_size,
    )
    assert_equal(joint_space_data, expected)
