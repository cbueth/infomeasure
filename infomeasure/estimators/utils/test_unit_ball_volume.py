"""Tests for the unit_ball_volume module."""

import pytest
from numpy import pi

from infomeasure.estimators.utils.unit_ball_volume import unit_ball_volume


@pytest.mark.parametrize(
    "d, p, r, expected",
    [
        (1, 2, 1, 2),  # d = 1, p = 2, r = 1
        (2, 2, 1, pi),  # d = 2, p = 2, r = 1
        (3, 2, 1, 4 / 3 * pi),  # d = 3, p = 2, r = 1
        (2, float("inf"), 1, 4),  # d = 2, p = inf, r = 1
        (2, 1, 1, 2),  # d = 2, p = 1, r = 1
        (2, 2, 2, 4 * pi),  # d = 2, p = 2, r = 2
        (3, 2, 2, 4 / 3 * pi * 8),  # d = 3, p = 2, r = 2
        (3, float("inf"), 2, 64),  # d = 3, p = inf, r = 2
        (3, 1, 2, 32 / 3),  # d = 3, p = 1, r = 2
    ],
)
def test_unit_ball_volume(d, p, r, expected):
    assert pytest.approx(unit_ball_volume(d, r, p), rel=1e-9) == expected
