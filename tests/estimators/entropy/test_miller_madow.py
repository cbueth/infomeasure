"""Explicit Miller-Madow entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy, estimator


@pytest.mark.parametrize(
    "data,base,expected",
    [
        ([1, 1, 1, 1, 1], 2, 0.0),  # K=1, N=5, correction=(1-1)/(2*5)=0
        ([1, 0, 1, 0], 2, 1.0 + (2 - 1) / (2 * 4) / log(2)),  # K=2, N=4, correction=1/8
        (
            ["a", 0, "a", 0],
            2,
            1.0 + (2 - 1) / (2 * 4) / log(2),
        ),  # K=2, N=4, correction=1/8
        (
            [[0, 3], [4, 4], [4, 4], [0, 3]],
            2,
            1.0 + (2 - 1) / (2 * 4) / log(2),
        ),  # K=2, N=4
        (
            [[0, 3], [4, 4], [4, 4], [3, 0]],
            2,
            1.5 + (3 - 1) / (2 * 4) / log(2),
        ),  # K=3, N=4
        (
            [1, 2, 3, 4, 5],
            2,
            2.321928094887362 + (5 - 1) / (2 * 5) / log(2),
        ),  # K=5, N=5
        (
            [1, 2, 3, 4, 5],
            10,
            0.6989700043360187 + (5 - 1) / (2 * 5) / log(10),
        ),  # K=5, N=5
        (
            [1, 2, 3, 4, 5],
            "e",
            1.6094379124341003 + (5 - 1) / (2 * 5),
        ),  # K=5, N=5, base=e
    ],
)
def test_miller_madow_entropy(data, base, expected):
    """Test the Miller-Madow entropy estimator."""
    assert entropy(data, approach="millermadow", base=base) == pytest.approx(expected)


@pytest.mark.parametrize(
    "data,base,expected",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2, 0.0),  # K=1, correction=0
        (
            ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
            2,
            0.0,
        ),  # K=1, correction=0
        (
            ([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]),
            2,
            3.0 + (8 - 1) / (2 * 8) / log(2),
        ),  # K=8, N=8
        (
            ([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]),
            10,
            0.90308999 + (8 - 1) / (2 * 8) / log(10),
        ),  # K=8, N=8
        (
            ([1, 2, 3 - 2] * 1000, [1, 2, 3 - 2] * 1000),
            10,
            0.2765069733573255,
        ),  # K=2, N=3000, joint combinations: (1,1), (2,2)
    ],
)
def test_miller_madow_joint_entropy(data, base, expected):
    """Test the Miller-Madow joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="millermadow", base=base)
    assert est.result() == pytest.approx(expected)


# try different bases with uniform distribution
@pytest.mark.parametrize("length", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("base", [2, 2.5, 3, 10, e])
def test_miller_madow_entropy_uniform(length, base):
    r"""Test the Miller-Madow entropy estimator with a uniform distribution.

    The entropy of a uniform distribution with Miller-Madow correction is:

    :math:`H_{MM}(X) = \log_b(n) + \frac{n-1}{2n \ln(b)}`
    """
    data = range(0, length)
    # Miller-Madow correction: (K-1)/(2N) where K=length, N=length
    correction = (length - 1) / (2 * length)
    if base == "e":
        expected = log(length) + correction
    else:
        expected = log(length) / log(base) + correction / log(base)

    assert entropy(data, approach="millermadow", base=base) == pytest.approx(expected)


@pytest.mark.parametrize(
    "data_p, data_q",
    [
        ([0, 1], [0, 1]),
        ([0, 1], [1, 0]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [2, 1, 0]),
    ],
)
def test_miller_madow_cross_entropy(data_p, data_q):
    """Test the Miller-Madow cross-entropy estimator."""
    assert entropy(data_p, data_q, approach="millermadow") > 0
