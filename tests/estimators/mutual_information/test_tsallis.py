"""Explicit tests for Tsallis mutual information."""

import pytest

from tests.conftest import generate_autoregressive_series_condition
from infomeasure.estimators.mutual_information import (
    TsallisMIEstimator,
    TsallisCMIEstimator,
)


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
    est = TsallisMIEstimator(data_x, data_y, k=k, q=q, base=2, noise_level=0)
    res = est.result()
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "data_x,data_y,cond,k,q,expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0, -0.14098512),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 1.1, 0.0344604586),
        (
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            4,
            1.0,
            -0.140985120,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            3,
            2.0,
            -0.15299879,
        ),
        (
            [1.0, 1.25, 0.91, 1.13, 1.32],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            1,
            1.1,
            -0.1974492227,
        ),
        (
            [1.01, 1.23, 0.92, 1.14, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            2,
            2.0,
            0.094036363822,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [11.08, 8.41, 11.47, 8.78, 14.09, 6.03, 10.67, 9.45, 12.72, 11.12],
            [7.95, -5.96, 7.03, -0.11, -1.86, 1.77, -2.49, 1.64, -3.05, -5.24],
            4,
            1.0,
            -0.04541147936,
        ),
    ],
)
def test_tsallis_cmi(data_x, data_y, cond, k, q, expected):
    """Test the conditional Tsallis mutual information estimator."""
    est = TsallisCMIEstimator(
        data_x, data_y, cond=cond, k=k, q=q, base=2, noise_level=0
    )
    res = est.result()
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,k,q,expected",
    (
        [5, 2, 1.0, 0.0708777],
        [5, 4, 1.0, 0.1378317],
        [5, 6, 1.0, 0.1805135],
        [5, 4, 1.1, 0.7758017],
        [5, 4, 1.0, 0.1378317],
        [5, 6, 1.1, 0.79445286],
        [6, 6, 1.1, 0.746171476],
        [7, 6, 1.1, 0.79626256],
    ),
)
def test_tsallis_cmi_autoregressive(rng_int, k, q, expected):
    """Test the Tsallis conditional mutual information estimator with autoregressive data."""
    data_x, data_y, cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = TsallisCMIEstimator(
        data_x, data_y, cond=cond, k=k, q=q, base=2, noise_level=0
    )
    assert est.result() == pytest.approx(expected)
