"""Test the TE slicing utility functions."""

import pytest
from numpy import arange, array, hstack, array_equal, column_stack
from numpy.random import default_rng
from numpy.testing import assert_equal

from infomeasure.estimators.utils.te_slicing import (
    te_observations,
    cte_observations,
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
    ) = te_observations(
        src,
        dest,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        step_size=step_size,
    )

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


@pytest.mark.parametrize(
    "source,dest,src_hist_len,dest_hist_len,step_size,expected",
    [
        (
            arange(10),
            arange(10, 20),
            3,
            2,
            2,
            (
                [
                    [0, 2, 4, 12, 14, 16],
                    [2, 4, 6, 14, 16, 18],
                ],
                [
                    [12, 14],
                    [14, 16],
                ],
                [
                    [0, 2, 4, 12, 14],
                    [2, 4, 6, 14, 16],
                ],
                [
                    [12, 14, 16],
                    [14, 16, 18],
                ],
            ),
        ),
        (
            arange(10).reshape(-1, 1),
            arange(10, 20).reshape(-1, 1),
            3,
            2,
            2,
            (
                [
                    [0, 2, 4, 12, 14, 16],
                    [2, 4, 6, 14, 16, 18],
                ],
                [
                    [12, 14],
                    [14, 16],
                ],
                [
                    [0, 2, 4, 12, 14],
                    [2, 4, 6, 14, 16],
                ],
                [
                    [12, 14, 16],
                    [14, 16, 18],
                ],
            ),
        ),
        (
            arange(10).reshape(-1, 1, 1),
            arange(10, 20).reshape(-1, 1, 1, 1),
            3,
            2,
            2,
            (
                [
                    [0, 2, 4, 12, 14, 16],
                    [2, 4, 6, 14, 16, 18],
                ],
                [
                    [12, 14],
                    [14, 16],
                ],
                [
                    [0, 2, 4, 12, 14],
                    [2, 4, 6, 14, 16],
                ],
                [
                    [12, 14, 16],
                    [14, 16, 18],
                ],
            ),
        ),
        (
            column_stack([arange(10), arange(10)]),
            column_stack([arange(10, 20), arange(10, 20)]),
            3,
            2,
            2,
            (
                [
                    [0, 0, 2, 2, 4, 4, 12, 12, 14, 14, 16, 16],
                    [2, 2, 4, 4, 6, 6, 14, 14, 16, 16, 18, 18],
                ],
                [
                    [12, 12, 14, 14],
                    [14, 14, 16, 16],
                ],
                [
                    [0, 0, 2, 2, 4, 4, 12, 12, 14, 14],
                    [2, 2, 4, 4, 6, 6, 14, 14, 16, 16],
                ],
                [
                    [12, 12, 14, 14, 16, 16],
                    [14, 14, 16, 16, 18, 18],
                ],
            ),
        ),
        (
            column_stack([arange(10), arange(10, 20)]),
            column_stack([arange(20, 30), arange(30, 40)]),
            3,
            2,
            2,
            (
                [
                    [0, 10, 2, 12, 4, 14, 22, 32, 24, 34, 26, 36],
                    [2, 12, 4, 14, 6, 16, 24, 34, 26, 36, 28, 38],
                ],
                [
                    [22, 32, 24, 34],
                    [24, 34, 26, 36],
                ],
                [
                    [0, 10, 2, 12, 4, 14, 22, 32, 24, 34],
                    [2, 12, 4, 14, 6, 16, 24, 34, 26, 36],
                ],
                [
                    [22, 32, 24, 34, 26, 36],
                    [24, 34, 26, 36, 28, 38],
                ],
            ),
        ),
        (  # mixed dimensions
            column_stack([arange(10), arange(10, 20)]),
            arange(20, 30),
            3,
            2,
            2,
            (
                [
                    [0, 10, 2, 12, 4, 14, 22, 24, 26],
                    [2, 12, 4, 14, 6, 16, 24, 26, 28],
                ],
                [
                    [22, 24],
                    [24, 26],
                ],
                [
                    [0, 10, 2, 12, 4, 14, 22, 24],
                    [2, 12, 4, 14, 6, 16, 24, 26],
                ],
                [
                    [22, 24, 26],
                    [24, 26, 28],
                ],
            ),
        ),
        (
            arange(10),
            column_stack([arange(20, 30), arange(30, 40)]),
            3,
            2,
            2,
            (
                [
                    [0, 2, 4, 22, 32, 24, 34, 26, 36],
                    [2, 4, 6, 24, 34, 26, 36, 28, 38],
                ],
                [
                    [22, 32, 24, 34],
                    [24, 34, 26, 36],
                ],
                [
                    [0, 2, 4, 22, 32, 24, 34],
                    [2, 4, 6, 24, 34, 26, 36],
                ],
                [
                    [22, 32, 24, 34, 26, 36],
                    [24, 34, 26, 36, 28, 38],
                ],
            ),
        ),
        (
            arange(10),
            column_stack([arange(20, 30), arange(30, 40)]),
            1,
            2,
            3,
            (
                [
                    [3, 20, 30, 23, 33, 26, 36],
                    [6, 23, 33, 26, 36, 29, 39],
                ],
                [
                    [20, 30, 23, 33],
                    [23, 33, 26, 36],
                ],
                [
                    [3, 20, 30, 23, 33],
                    [6, 23, 33, 26, 36],
                ],
                [
                    [20, 30, 23, 33, 26, 36],
                    [23, 33, 26, 36, 29, 39],
                ],
            ),
        ),
        (
            column_stack([arange(20, 30), arange(30, 40)]),
            arange(10),
            2,
            1,
            1,
            (
                [
                    [20, 30, 21, 31, 1, 2],
                    [21, 31, 22, 32, 2, 3],
                    [22, 32, 23, 33, 3, 4],
                    [23, 33, 24, 34, 4, 5],
                    [24, 34, 25, 35, 5, 6],
                    [25, 35, 26, 36, 6, 7],
                    [26, 36, 27, 37, 7, 8],
                    [27, 37, 28, 38, 8, 9],
                ],
                [[1], [2], [3], [4], [5], [6], [7], [8]],
                [
                    [20, 30, 21, 31, 1],
                    [21, 31, 22, 32, 2],
                    [22, 32, 23, 33, 3],
                    [23, 33, 24, 34, 4],
                    [24, 34, 25, 35, 5],
                    [25, 35, 26, 36, 6],
                    [26, 36, 27, 37, 7],
                    [27, 37, 28, 38, 8],
                ],
                [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]],
            ),
        ),
    ],
)
def test_te_observations_explicit(
    source, dest, src_hist_len, dest_hist_len, step_size, expected
):
    """Test the TE observations data arrays explicitly."""
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = te_observations(source, dest, src_hist_len, dest_hist_len, step_size)
    assert array_equal(joint_space_data, expected[0])
    assert array_equal(data_dest_past_embedded, expected[1])
    assert array_equal(marginal_1_space_data, expected[2])
    assert array_equal(marginal_2_space_data, expected[3])


@pytest.mark.parametrize(
    "source,dest,src_hist_len,dest_hist_len,step_size,expected",
    [
        (
            arange(5),
            arange(10, 15),
            1,
            1,
            1,
            (
                [[0], [1], [2], [3]],  # x
                [[10], [11], [12], [13]],  # y
                [[11], [12], [13], [14]],  # y'
            ),
        ),
        (
            arange(5),
            arange(10, 15),
            1,
            1,
            1,
            ([[0], [1], [2], [3]], [[10], [11], [12], [13]], [[11], [12], [13], [14]]),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            1,
            1,
            2,
            (
                [[0], [2]],
                [[10, 15], [12, 17]],
                [[12, 17], [14, 19]],
            ),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            1,
            2,
            2,
            ([[2]], [[10, 15, 12, 17]], [[14, 19]]),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            1,
            1,
            2,
            ([[0], [2]], [[10, 15], [12, 17]], [[12, 17], [14, 19]]),
        ),
        (
            arange(20).reshape(-1, 1),
            arange(20, 40),
            3,
            2,
            3,
            (
                [[0, 3, 6], [3, 6, 9], [6, 9, 12], [9, 12, 15]],
                [[23, 26], [26, 29], [29, 32], [32, 35]],
                [[29], [32], [35], [38]],
            ),
        ),
    ],
)
def test_te_observations_explicit_separate(
    source, dest, src_hist_len, dest_hist_len, step_size, expected
):
    """Test the TE observations data arrays with spaces not constructed"""
    (
        src_history,
        dest_history,
        dest_future,
    ) = te_observations(
        source,
        dest,
        src_hist_len,
        dest_hist_len,
        step_size,
        construct_joint_spaces=False,
    )
    assert array_equal(src_history, expected[0])
    assert array_equal(dest_history, expected[1])
    assert array_equal(dest_future, expected[2])


@pytest.mark.parametrize(
    "source,dest,cond,src_hist_len,dest_hist_len,cond_hist_len,step_size,expected",
    [
        (
            arange(5),
            arange(10, 15),
            arange(20, 25),
            1,
            1,
            1,
            1,
            (
                [  # x,  y, y',  z
                    [0, 10, 11, 20],
                    [1, 11, 12, 21],
                    [2, 12, 13, 22],
                    [3, 13, 14, 23],
                ],
                [  #  y,  z
                    [10, 20],
                    [11, 21],
                    [12, 22],
                    [13, 23],
                ],
                [  # x,  y,  z
                    [0, 10, 20],
                    [1, 11, 21],
                    [2, 12, 22],
                    [3, 13, 23],
                ],
                [  #  y, y',  z
                    [10, 11, 20],
                    [11, 12, 21],
                    [12, 13, 22],
                    [13, 14, 23],
                ],
            ),
        ),
        (
            arange(5),
            arange(10, 15),
            arange(20, 25),
            1,
            1,
            2,
            1,
            (
                [  # x,  y, y', z1, z2
                    [1, 11, 12, 20, 21],
                    [2, 12, 13, 21, 22],
                    [3, 13, 14, 22, 23],
                ],
                [  #  y, z1, z2
                    [11, 20, 21],
                    [12, 21, 22],
                    [13, 22, 23],
                ],
                [  # x,  y, z1, z2
                    [1, 11, 20, 21],
                    [2, 12, 21, 22],
                    [3, 13, 22, 23],
                ],
                [  #  y, y', z1, z2
                    [11, 12, 20, 21],
                    [12, 13, 21, 22],
                    [13, 14, 22, 23],
                ],
            ),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            arange(20, 25),
            1,
            1,
            1,
            2,
            (
                [  # x, ( y ) , ( y') ,  z
                    [0, 10, 15, 12, 17, 20],
                    [2, 12, 17, 14, 19, 22],
                ],
                [  # ( y ) ,  z
                    [10, 15, 20],
                    [12, 17, 22],
                ],
                [  # x, ( y ) ,  z
                    [0, 10, 15, 20],
                    [2, 12, 17, 22],
                ],
                [  # ( y ) , ( y') ,  z
                    [10, 15, 12, 17, 20],
                    [12, 17, 14, 19, 22],
                ],
            ),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            arange(20, 25),
            1,
            2,
            1,
            2,
            (
                [  # x, ( y1 ), ( y2 ), ( y') ,  z
                    [2, 10, 15, 12, 17, 14, 19, 22],
                ],
                [  # ( y1 ), ( y2 ),  z
                    [10, 15, 12, 17, 22],
                ],
                [  # x, ( y1 ), ( y2 ),  z
                    [2, 10, 15, 12, 17, 22],
                ],
                [  # ( y1 ), ( y2 ), ( y') ,  z
                    [10, 15, 12, 17, 14, 19, 22],
                ],
            ),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            arange(20, 25),
            1,
            1,
            2,
            2,
            (
                [  # x, ( y ) , ( y') , z1, z2
                    [2, 12, 17, 14, 19, 20, 22],
                ],
                [  # ( y ) , z1, z2
                    [12, 17, 20, 22],
                ],
                [  # x, ( y ) , z1, z2
                    [2, 12, 17, 20, 22],
                ],
                [  # ( y ) , ( y') , z1, z2
                    [12, 17, 14, 19, 20, 22],
                ],
            ),
        ),
        (
            arange(20).reshape(-1, 1),
            arange(20, 40),
            column_stack([arange(40, 60), arange(60, 80), arange(80, 100)]),
            3,
            2,
            1,
            3,
            (
                [
                    [0, 3, 6, 23, 26, 29, 46, 66, 86],
                    [3, 6, 9, 26, 29, 32, 49, 69, 89],
                    [6, 9, 12, 29, 32, 35, 52, 72, 92],
                    [9, 12, 15, 32, 35, 38, 55, 75, 95],
                ],
                [
                    [23, 26, 46, 66, 86],
                    [26, 29, 49, 69, 89],
                    [29, 32, 52, 72, 92],
                    [32, 35, 55, 75, 95],
                ],
                [
                    [0, 3, 6, 23, 26, 46, 66, 86],
                    [3, 6, 9, 26, 29, 49, 69, 89],
                    [6, 9, 12, 29, 32, 52, 72, 92],
                    [9, 12, 15, 32, 35, 55, 75, 95],
                ],
                [
                    [23, 26, 29, 46, 66, 86],
                    [26, 29, 32, 49, 69, 89],
                    [29, 32, 35, 52, 72, 92],
                    [32, 35, 38, 55, 75, 95],
                ],
            ),
        ),
    ],
)
def test_cte_observations_explicit(
    source, dest, cond, src_hist_len, dest_hist_len, cond_hist_len, step_size, expected
):
    """Test the CTE observations data arrays explicitly."""
    (
        joint_space_data,
        data_dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    ) = cte_observations(
        source, dest, cond, src_hist_len, dest_hist_len, cond_hist_len, step_size
    )
    assert array_equal(joint_space_data, expected[0])
    assert array_equal(data_dest_past_embedded, expected[1])
    assert array_equal(marginal_1_space_data, expected[2])
    assert array_equal(marginal_2_space_data, expected[3])


@pytest.mark.parametrize(
    "source,dest,cond,src_hist_len,dest_hist_len,cond_hist_len,step_size,expected",
    [
        (
            arange(5),
            arange(10, 15),
            arange(20, 25),
            1,
            1,
            1,
            1,
            (
                [[0], [1], [2], [3]],  # x
                [[10], [11], [12], [13]],  # y
                [[11], [12], [13], [14]],  # y'
                [[20], [21], [22], [23]],  # z
            ),
        ),
        (
            arange(5),
            arange(10, 15),
            arange(20, 25),
            1,
            1,
            2,
            1,
            (
                [[1], [2], [3]],
                [[11], [12], [13]],
                [[12], [13], [14]],
                [[20, 21], [21, 22], [22, 23]],
            ),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            arange(20, 25),
            1,
            1,
            1,
            2,
            ([[0], [2]], [[10, 15], [12, 17]], [[12, 17], [14, 19]], [[20], [22]]),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            arange(20, 25),
            1,
            2,
            1,
            2,
            ([[2]], [[10, 15, 12, 17]], [[14, 19]], [[22]]),
        ),
        (
            arange(5),
            column_stack([arange(10, 15), arange(15, 20)]),
            arange(20, 25),
            1,
            1,
            2,
            2,
            ([[2]], [[12, 17]], [[14, 19]], [[20, 22]]),
        ),
        (
            arange(20).reshape(-1, 1),
            arange(20, 40),
            column_stack([arange(40, 60), arange(60, 80), arange(80, 100)]),
            3,
            2,
            1,
            3,
            (
                [[0, 3, 6], [3, 6, 9], [6, 9, 12], [9, 12, 15]],
                [[23, 26], [26, 29], [29, 32], [32, 35]],
                [[29], [32], [35], [38]],
                [[46, 66, 86], [49, 69, 89], [52, 72, 92], [55, 75, 95]],
            ),
        ),
    ],
)
def test_cte_observations_explicit_separate(
    source, dest, cond, src_hist_len, dest_hist_len, cond_hist_len, step_size, expected
):
    """Test the CTE observations data arrays with spaces not constructed"""
    (
        src_history,
        dest_history,
        dest_future,
        cond_history,
    ) = cte_observations(
        source,
        dest,
        cond,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        step_size=step_size,
        construct_joint_spaces=False,
    )
    assert array_equal(src_history, expected[0])
    assert array_equal(dest_history, expected[1])
    assert array_equal(dest_future, expected[2])
    assert array_equal(cond_history, expected[3])


@pytest.mark.parametrize(
    "replace_idx,value,expected_error",
    [
        (3, 0, ValueError),
        (3, None, ValueError),
        (3, 1.0, ValueError),
        (4, 0, ValueError),
        (4, -3, ValueError),
        (5, 0, ValueError),
        (5, "0", ValueError),
        (6, 0, ValueError),
        (0, 1, TypeError),
        (1, [1], TypeError),
        (1, 1, TypeError),
        (2, None, TypeError),
    ],
)
def test_cte_observations_invalid_inputs(replace_idx, value, expected_error):
    """Test the CTE observations data arrays with invalid inputs."""
    arr = arange(10)
    args = [arr, arr, arr, 1, 1, 1, 1]
    args[replace_idx] = value
    with pytest.raises(expected_error):
        cte_observations(*tuple(args))


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
                [2, 2, 5, 5, 6, 6, 7, 7],
                [3, 3, 6, 6, 7, 7, 8, 8],
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
@pytest.mark.parametrize("value", [True, default_rng(5378)])
@pytest.mark.parametrize("argument", ["permute_src", "resample_src"])
def test_te_observations_permute_resample_src(
    src_hist_len, dest_hist_len, step_size, value, argument
):
    """Test the TE observations data arrays with permutation and resampling."""
    source = arange(100)
    destination = arange(100, 200)
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
        **{argument: value},  # permute_src or resample_src with True or Generator
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


def test_te_observations_permute_resample_exclusivity():
    """Test TE observations error when setting both
    ``permute_src`` and ``resample_src``."""
    with pytest.raises(ValueError):
        te_observations(
            arange(10),
            arange(10, 20),
            src_hist_len=1,
            dest_hist_len=1,
            step_size=1,
            permute_src=True,
            resample_src=True,
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
