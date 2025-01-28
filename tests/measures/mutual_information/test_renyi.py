"""Explicit tests for Renyi mutual information."""

import pytest

from infomeasure.measures.mutual_information import RenyiMIEstimator


@pytest.mark.parametrize(
    "data_x,data_y,k,alpha,expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0, 0.3235175076367409),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.1, 0.3710421557800414),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 4, 1.0, -0.46369086049188546),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            3,
            2.0,
            0.209995684452404,
        ),
        (
            [1.0, 1.25, 0.91, 1.13, 1.32],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            1,
            1.1,
            1.6050136160482718,
        ),
        (
            [1.01, 1.23, 0.92, 1.14, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            2,
            2.0,
            1.1073653229227254,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [11.08, 8.41, 11.47, 8.78, 14.09, 6.03, 10.67, 9.45, 12.72, 11.12],
            4,
            1.0,
            0.15135562730632035,
        ),
    ],
)
def test_renyi_mi(data_x, data_y, k, alpha, expected):
    """Test the Renyi mutual information estimator."""
    est = RenyiMIEstimator(data_x, data_y, k=k, alpha=alpha)
    res = est.results()
    assert res == pytest.approx(expected)
