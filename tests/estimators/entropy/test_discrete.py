"""Explicit discrete entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy, estimator


@pytest.mark.parametrize(
    "data,base,expected",
    [
        ([1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], 2, 1.0),
        (["a", 0, "a", 0], 2, 1.0),
        ([[0, 3], [4, 4], [4, 4], [0, 3]], 2, 1.0),
        ([[0, 3], [4, 4], [4, 4], [3, 0]], 2, 1.5),
        ([1, 2, 3, 4, 5], 2, 2.321928094887362),
        ([1, 2, 3, 4, 5], 10, 0.6989700043360187),
        ([1, 2, 3, 4, 5], "e", 1.6094379124341003),
    ],
)
def test_discrete_entropy(data, base, expected):
    """Test the discrete entropy estimator."""
    assert entropy(data, approach="discrete", base=base) == pytest.approx(expected)


@pytest.mark.parametrize(
    "data,base,expected",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2, 0.0),
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2, 0.0),
        (([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]), 2, 3.0),
        (([1, 1, 7, 2, 3, 6, 6, 3], [2, 3, 6, 6, 3, 6, 5, 7]), 10, 0.90308999),
        (([1, 2, 3 - 2] * 1000, [1, 2, 3 - 2] * 1000), 10, 0.276434591),
    ],
)
def test_discrete_joint_entropy(data, base, expected):
    """Test the discrete joint entropy estimator."""
    est = estimator(data, measure="entropy", approach="discrete", base=base)
    assert est.result() == pytest.approx(expected)
    est.local_vals()


# try different bases with uniform distribution
@pytest.mark.parametrize("length", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("base", [2, 2.5, 3, 10, e])
def test_discrete_entropy_uniform(length, base):
    r"""Test the discrete entropy estimator with a uniform distribution.

    The entropy of a uniform distribution is given by:

    :math:`H(X) = -\log_b(1/n) = \log_b(n)`
    """
    data = range(0, length)
    assert entropy(data, approach="discrete", base=base) == pytest.approx(
        log(length) / log(base)
    )


@pytest.mark.parametrize(
    "data_p, data_q",
    [
        ([0, 1], [0, 1]),
        ([0, 1], [1, 0]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [2, 1, 0]),
    ],
)
def test_discrete_cross_entropy(data_p, data_q):
    """Test the discrete cross-entropy estimator."""
    assert entropy(data_p, data_q, approach="discrete") > 0
