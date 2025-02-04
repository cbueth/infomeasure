"""Explicit discrete entropy estimator tests."""

import pytest
from numpy import e, log

from infomeasure import entropy


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
