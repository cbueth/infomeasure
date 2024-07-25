"""Test the TE slicing utility functions."""

import pytest
from numpy import arange, array

from infomeasure.measures.utils.te_slicing import (
    construct_embedded_data_vectorized,
    te_observations,
)


@pytest.mark.parametrize(
    "data, step_size, emb_dim, other_emb_dim, expected",
    [
        (arange(10), 1, 1, None, arange(10).reshape(-1, 1)),
        (
            arange(10),
            1,
            2,
            None,
            array(
                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
            ),
        ),
        (
            arange(10),
            2,
            2,
            None,
            array([[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9]]),
        ),
        (arange(10), 4, 3, None, array([[0, 4, 8], [1, 5, 9]])),
        (arange(10), 4, 4, None, array([])),
        (
            arange(10),
            4,
            2,
            None,
            array([[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]),
        ),
        (arange(10), 1, 10, None, array([arange(10)])),
        (arange(10), 9, 2, None, array([[0, 9]])),
        (arange(10), 1, 1, 1, arange(10).reshape(-1, 1)),
        (arange(10), 1, 1, 2, arange(9).reshape(-1, 1)),
        (arange(10), 1, 1, 5, arange(6).reshape(-1, 1)),
        (
            arange(10),
            1,
            5,
            2,
            array(
                [
                    [0, 1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7],
                    [4, 5, 6, 7, 8],
                    [5, 6, 7, 8, 9],
                ]
            ),
        ),
        (array([]), 1, 1, None, array([])),
    ],
)
def test_construct_embedded_data_vectorized(
    data, step_size, emb_dim, other_emb_dim, expected
):
    """Test the construction of embedded data."""
    result = construct_embedded_data_vectorized(data, step_size, emb_dim, other_emb_dim)
    assert result.shape == expected.shape
    assert (result == expected).all()


@pytest.mark.parametrize("data_len", [1, 2, 10, 100, 1e5])
@pytest.mark.parametrize("src_hist_len", [1, 2, 3])
@pytest.mark.parametrize("dest_hist_len", [1, 2, 3])
@pytest.mark.parametrize("step_size", [1, 2, 3])
def test_te_observations_shape(data_len, src_hist_len, dest_hist_len, step_size):
    """Test the shape of the TE observations data arrays."""
    src = dest = arange(data_len)
    # max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * step_size
    # max_len cannot be negative
    if data_len - (max(src_hist_len, dest_hist_len) - 1) * step_size <= 0:
        with pytest.raises(ValueError):
            te_observations(src, dest, src_hist_len, dest_hist_len, step_size)
        return
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(src, dest, src_hist_len, dest_hist_len, step_size)
    max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * step_size
    assert joint_space_data.shape == (max_len, src_hist_len + dest_hist_len + 1)
    assert data_dest_past_embedded.shape == (max_len, dest_hist_len)
    assert marginal_1_space_data.shape == (max_len, dest_hist_len + src_hist_len)
    assert marginal_2_space_data.shape == (max_len, dest_hist_len + 1)


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
                [14, 10, 12, 0, 2, 4],
                [15, 11, 13, 1, 3, 5],
                [16, 12, 14, 2, 4, 6],
                [17, 13, 15, 3, 5, 7],
                [18, 14, 16, 4, 6, 8],
                [19, 15, 17, 5, 7, 9],
            ]
        )
        == joint_space_data
    ).all()
    assert (
        array([[10, 12], [11, 13], [12, 14], [13, 15], [14, 16], [15, 17]])
        == data_dest_past_embedded
    ).all()
    assert (
        array(
            [
                [10, 12, 0, 2, 4],
                [11, 13, 1, 3, 5],
                [12, 14, 2, 4, 6],
                [13, 15, 3, 5, 7],
                [14, 16, 4, 6, 8],
                [15, 17, 5, 7, 9],
            ]
        )
        == marginal_1_space_data
    ).all()
    assert (
        array(
            [
                [14, 10, 12],
                [15, 11, 13],
                [16, 12, 14],
                [17, 13, 15],
                [18, 14, 16],
                [19, 15, 17],
            ]
        )
        == marginal_2_space_data
    ).all()
