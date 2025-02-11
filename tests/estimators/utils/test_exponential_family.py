"""Test exponential family helper functions."""

import pytest
from numpy import array, pi, inf

from infomeasure.estimators.utils.exponential_family import (
    calculate_common_entropy_components,
)


@pytest.mark.parametrize(
    "data,k,V_m, rho_k, N, m",
    [
        ([[1.0], [2.0]], 1, 2.0, [1.0, 1.0], 2, 1),
        ([[1.0], [2.0], [3.0]], 1, 2.0, [1.0, 1.0, 1.0], 3, 1),
        ([[1.0], [2.0], [3.0]], 2, 2.0, [2.0, 1.0, 2.0], 3, 1),
        ([[1], [2], [3]], 2, 2.0, [2.0, 1.0, 2.0], 3, 1),
        (
            [[1.0, 2.0], [2.0, 3.0]],
            1,
            pi,
            [2**0.5, 2**0.5],
            2,
            2,
        ),
        (
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            1,
            4.188790,
            [3**0.5, 3**0.5],
            2,
            3,
        ),
    ],
)
def test_calculate_common_entropy_components(data, k, V_m, rho_k, N, m):
    """Test the common entropy components calculation."""
    data = array(data)
    res = calculate_common_entropy_components(data, k)
    assert res[0] == pytest.approx(V_m)
    assert res[1] == pytest.approx(rho_k)
    assert res[2] == pytest.approx(N)
    assert res[3] == pytest.approx(m)


@pytest.mark.parametrize(
    "data,k",
    [
        ([[0.0]], 1),
        ([[1.0], [2.0], [3.0]], 3),
        ([[1.0], [2.0], [3.0]], 4),
        (
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            3,
        ),
    ],
)
def test_calculate_common_entropy_components_k_too_large(data, k):
    """Test the common entropy components calculation with k too large."""
    data = array(data)
    with pytest.raises(ValueError, match="number of nearest neighbors must be smaller"):
        calculate_common_entropy_components(data, k)
