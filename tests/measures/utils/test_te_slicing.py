"""Test the TE slicing utility functions."""

import pytest
from numpy import arange, array, hstack, array_equal

from infomeasure.measures.utils.te_slicing import (
    te_observations,
)


@pytest.mark.parametrize("data_len", [1, 2, 10, 100, 1e5])
@pytest.mark.parametrize("src_hist_len", [1, 2, 3])
@pytest.mark.parametrize("dest_hist_len", [1, 2, 3])
@pytest.mark.parametrize("step_size", [1, 2, 3])
def test_te_observations_old_implementation(
    data_len, src_hist_len, dest_hist_len, step_size
):
    """Test the shape of the TE observations data arrays.

    Compare output to old implementation.
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
    max_delay = max(dest_hist_len * step_size, src_hist_len * step_size)
    x_future = array([src[i + max_delay] for i in range(n - max_delay)])
    x_history = array(
        [src[i - src_hist_len * step_size : i : step_size] for i in range(max_delay, n)]
    )
    y_history = array(
        [
            dest[i - dest_hist_len * step_size : i : step_size]
            for i in range(max_delay, n)
        ]
    )
    assert array_equal(
        joint_space_data, hstack((x_future.reshape(-1, 1), x_history, y_history))
    )
    assert array_equal(data_dest_past_embedded, x_history)
    assert array_equal(marginal_1_space_data, hstack((x_history, y_history)))
    assert array_equal(
        marginal_2_space_data, hstack((x_future.reshape(-1, 1), x_history))
    )


def test_te_observations():
    """Test the TE observations data arrays explicitly."""
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(
        arange(10), arange(10, 20), src_hist_len=2, dest_hist_len=3, step_size=2
    )
    assert (
        array(
            [
                [6, 2, 4, 10, 12, 14],
                [7, 3, 5, 11, 13, 15],
                [8, 4, 6, 12, 14, 16],
                [9, 5, 7, 13, 15, 17],
            ]
        )
        == joint_space_data
    ).all()
    assert (array([[2, 4], [3, 5], [4, 6], [5, 7]]) == data_dest_past_embedded).all()
    assert (
        array(
            [
                [2, 4, 10, 12, 14],
                [3, 5, 11, 13, 15],
                [4, 6, 12, 14, 16],
                [5, 7, 13, 15, 17],
            ]
        )
        == marginal_1_space_data
    ).all()
    assert (
        array([[6, 2, 4], [7, 3, 5], [8, 4, 6], [9, 5, 7]]) == marginal_2_space_data
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
