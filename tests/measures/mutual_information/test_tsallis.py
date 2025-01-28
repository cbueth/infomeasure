"""Explicit tests for Tsallis mutual information."""

import pytest

from infomeasure.measures.mutual_information import TsallisMIEstimator


@pytest.mark.parametrize(
    "data_x,data_y,k,q,expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0, 0.3235175076367409),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.1, 0.36793161147482856),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 4, 1.0, -0.46369086049188546),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            3,
            2.0,
            0.19083137735244038,
        ),
        (
            [1.0, 1.25, 0.91, 1.13, 1.32],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            1,
            1.1,
            1.1061468907268268,
        ),
        (
            [1.01, 1.23, 0.92, 1.14, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            2,
            2.0,
            1.0763337324321895,
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
def test_tsallis_mi(data_x, data_y, k, q, expected):
    """Test the Tsallis mutual information estimator."""
    est = TsallisMIEstimator(data_x, data_y, k=k, q=q)
    res = est.results()
    assert res == pytest.approx(expected)
