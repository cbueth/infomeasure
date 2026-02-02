"""Explicit Kraskov-Stoegbauer-Grassberger mutual information estimator tests."""

import numpy as np
import pytest
from numpy import linspace, inf, ndarray

from infomeasure import mutual_information
from infomeasure.estimators.mutual_information import KSGMIEstimator, KSGCMIEstimator
from tests.conftest import generate_autoregressive_series_condition


@pytest.mark.parametrize(
    "data_x,data_y,k,ksg_id,minkowski_p,expected",
    [
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 1, 1, -0.25),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 1, 2, -0.25),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 2, 2, -0.5),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 1, 3, -0.25),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            1,
            1,
            2,
            -0.8500000000000002,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            2,
            1,
            2,
            -0.1833333333333334,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            3,
            1,
            2,
            -0.48333333333333306,
        ),
        (
            [1.0, 1.25, 0.91, 1.13, 1.32],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            1,
            1,
            inf,
            -0.01666666666666694,
        ),
        (
            [1.01, 1.23, 0.92, 1.14, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            2,
            1,
            inf,
            0.21666666666666648,
        ),
        (
            [1.04, 1.23, 0.92, 1.1, 1.34],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            3,
            1,
            inf,
            0.11666666666666678,
        ),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 1, 1.852972755734859),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 2, 2.828044184306288),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 3, 2.8360441843062887),
        (linspace(0, 1, 100), linspace(0, 1, 100), 1, 1, 2, 2.1973775176396204),
        (linspace(0, 1, 100), linspace(0, 1, 100), 2, 1, 2, 3.177377517639621),
        (linspace(0, 1, 100), linspace(0, 1, 100), 3, 1, 2, 2.520710850972955),
        (linspace(0, 1, 100), linspace(1, 0, 100), 3, 1, 2, 2.520710850972955),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [-8.59, 8.41, 3.76, 3.77, 5.69, 1.75, -3.2, -4.0, -4.0, 6.85],
            4,
            1,
            inf,
            -0.034126984126984186,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [11.08, 8.41, 11.47, 8.78, 14.09, 6.03, 10.67, 9.45, 12.72, 11.12],
            4,
            1,
            inf,
            0.0816269841269841,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            4,
            1,
            inf,
            0.9956349206349205,
        ),
        (
            [0.32, -0.89, -0.01, 1.56, 0.24, 1.78, -1.63, 0.82, 0.12, 2.29],
            [0.22, -0.99, -0.11, 1.46, 0.14, 1.68, -1.73, 0.72, 0.02, 2.19],
            4,
            1,
            inf,
            0.9206349206349206,
        ),
        (
            [1.8, 1.1, 1.8, 1.1, 1.8, -2.0, 1.1, 1.8, -2.0, 1.1],
            [-0.6, 0.4, -0.6, -0.6, -1.3, -1.3, -1.3, -0.6, -0.6, 0.4],
            4,
            1,
            inf,
            0.48888888888888876,
        ),
    ],
)
def test_ksg_mi(data_x, data_y, k, ksg_id, minkowski_p, expected):
    """Test the Kraskov-Stoegbauer-Grassberger mutual information estimator."""
    est = KSGMIEstimator(
        data_x,
        data_y,
        k=k,
        ksg_id=ksg_id,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "data_x,data_y,k,ksg_id,minkowski_p,base,expected",
    [
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            4,
            1,
            3,
            2,
            -0.36067376022224085,
        ),
        (
            [1.04, 1.23, 0.92, 1.1, 1.34],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            3,
            1,
            inf,
            2,
            0.1683144214370459,
        ),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 3, 2, 4.091546880440667),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [-8.59, 8.41, 3.76, 3.77, 5.69, 1.75, -3.2, -4.0, -4.0, 6.85],
            4,
            1,
            inf,
            2,
            -0.049234830760496444,
        ),
    ],
)
def test_ksg_mi_base(data_x, data_y, k, ksg_id, minkowski_p, base, expected):
    """Test the Kraskov-Stoegbauer-Grassberger mutual information estimator with a
    different base."""
    est = KSGMIEstimator(
        data_x,
        data_y,
        k=k,
        ksg_id=ksg_id,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "data_x,data_y,k,ksg_id,minkowski_p,expected",
    [
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 1, 1, -0.25),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 1, 2, -0.25),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 4, 1, 3, -0.25),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 1, 1, 2, -0.85),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 2, 1, 2, -0.1833333),
        ([1.0, 1.2, 0.9, 1.1, 1.3], [1.3, 1.1, 0.9, 1.2, 1.0], 3, 1, 2, -0.4833333),
        (
            [1.0, 1.25, 0.91, 1.13, 1.32],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            1,
            1,
            inf,
            -0.0166666666,
        ),
        (
            [1.01, 1.23, 0.92, 1.14, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            2,
            1,
            inf,
            0.216666666,
        ),
        (
            [1.04, 1.23, 0.92, 1.1, 1.34],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            3,
            1,
            inf,
            1.7763568e-16,
        ),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 1, 1.85297275573),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 2, 2.828044184306288),
        (linspace(0, 1, 100), linspace(0, 1, 100), 4, 1, 3, 2.8360441843062887),
        (linspace(0, 1, 100), linspace(0, 1, 100), 1, 1, 2, 2.1973775176396204),
        (linspace(0, 1, 100), linspace(0, 1, 100), 2, 1, 2, 3.177377517639621),
        (linspace(0, 1, 100), linspace(0, 1, 100), 3, 1, 2, 2.520710850972955),
        (linspace(0, 1, 100), linspace(1, 0, 100), 3, 1, 2, 2.520710850972955),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [-8.59, 8.41, 3.76, 3.77, 5.69, 1.75, -3.2, -4.0, -4.0, 6.85],
            4,
            1,
            inf,
            -0.0734126984,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [11.08, 8.41, 11.47, 8.78, 14.09, 6.03, 10.67, 9.45, 12.72, 11.12],
            4,
            1,
            inf,
            0.226547619,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            4,
            1,
            inf,
            0.9956349206349205,
        ),
        (
            [0.32, -0.89, -0.01, 1.56, 0.24, 1.78, -1.63, 0.82, 0.12, 2.29],
            [0.22, -0.99, -0.11, 1.46, 0.14, 1.68, -1.73, 0.72, 0.02, 2.19],
            4,
            1,
            inf,
            0.8456349206349205,
        ),
        (
            [1.8, 1.1, 1.8, 1.1, 1.8, -2.0, 1.1, 1.8, -2.0, 1.1],
            [-0.6, 0.4, -0.6, -0.6, -1.3, -1.3, -1.3, -0.6, -0.6, 0.4],
            4,
            1,
            inf,
            0.3841666666,
        ),
    ],
)
def test_ksg_mi_normalized(data_x, data_y, k, ksg_id, minkowski_p, expected):
    """
    Test the Kraskov-Stoegbauer-Grassberger mutual information estimator with normalization.
    """
    est = KSGMIEstimator(
        data_x,
        data_y,
        k=k,
        ksg_id=ksg_id,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        normalize=True,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "data_x,data_y,cond,k,ksg_id,minkowski_p,base,expected",
    [
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            4,
            1,
            1,
            "e",
            -0.25,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            4,
            1,
            2,
            "e",
            -0.25,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            4,
            2,
            2,
            "e",
            -0.5,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            4,
            1,
            3,
            "e",
            -0.25,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            1,
            1,
            2,
            "e",
            -1.1333333333333333,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            2,
            1,
            2,
            "e",
            -0.7666666666666664,
        ),
        (
            [1.0, 1.2, 0.9, 1.1, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            3,
            1,
            2,
            "e",
            -0.4833333333333331,
        ),
        (
            [1.0, 1.25, 0.91, 1.13, 1.32],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            1,
            1,
            inf,
            "e",
            0.13333333333333322,
        ),
        (
            [1.01, 1.23, 0.92, 1.14, 1.3],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            2,
            1,
            inf,
            "e",
            0.016666666666666587,
        ),
        (
            [1.04, 1.23, 0.92, 1.1, 1.34],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            3,
            1,
            inf,
            "e",
            0.19999999999999996,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            4,
            1,
            1,
            "e",
            0.1810945165945167,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            4,
            1,
            2,
            "e",
            0.1133333333333334,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            4,
            1,
            3,
            "e",
            -0.24200000000000002,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            1,
            1,
            2,
            "e",
            -1.49,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            2,
            1,
            2,
            "e",
            -0.49333333333333335,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            3,
            1,
            2,
            "e",
            -0.21066666666666634,
        ),
        (
            linspace(0, 1, 100),
            linspace(0, 1, 100),
            linspace(1, 0, 100),
            3,
            1,
            2,
            "e",
            -0.21066666666666634,
        ),
        (
            linspace(0, 1, 100),
            linspace(1, 0, 100),
            linspace(1, 0, 100),
            3,
            1,
            2,
            "e",
            -0.21066666666666634,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [7.95, -5.96, 7.03, -0.11, -1.86, 1.77, -2.49, 1.64, -3.05, -5.24],
            [-8.59, 8.41, 3.76, 3.77, 5.69, 1.75, -3.2, -4.0, -4.0, 6.85],
            4,
            1,
            inf,
            "e",
            -0.061309523809523855,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [7.95, -5.96, 7.03, -0.11, -1.86, 1.77, -2.49, 1.64, -3.05, -5.24],
            [11.08, 8.41, 11.47, 8.78, 14.09, 6.03, 10.67, 9.45, 12.72, 11.12],
            4,
            1,
            inf,
            "e",
            -0.09285714285714297,
        ),
        (
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            [7.95, -5.96, 7.03, -0.11, -1.86, 1.77, -2.49, 1.64, -3.05, -5.24],
            [-2.49, 1.64, -3.05, 7.95, -5.96, 1.77, -5.24, 7.03, -0.11, -1.86],
            4,
            1,
            inf,
            "e",
            0.0,
        ),
        (
            [0.32, -0.89, -0.01, 1.56, 0.24, 1.78, -1.63, 0.82, 0.12, 2.29],
            [0.22, -0.99, -0.11, 1.46, 0.14, 1.68, -1.73, 0.72, 0.02, 2.19],
            [0.20, -0.97, 0.11, 1.16, 0.14, 1.18, -1.43, 0.72, 0.02, 2.29],
            4,
            1,
            inf,
            "e",
            0.25857142857142856,
        ),
        (
            [1.8, 1.1, 1.8, 1.1, 1.8, -2.0, 1.1, 1.8, -2.0, 1.1],
            [-0.6, 0.4, -0.6, -0.6, -1.3, -1.3, -1.3, -0.6, -0.6, 0.4],
            [-0.6, 0.4, -0.6, -0.6, -1.3, -1.3, -1.3, -0.6, -0.6, 0.4],
            4,
            1,
            inf,
            "e",
            0.49999999999999983,
        ),
        (
            [1.8, 1.1, 1.8, 1.1, 1.8, -2.0, 1.1, 1.8, -2.0, 1.1],
            [-0.6, 0.4, -0.6, -0.6, -1.3, -1.3, -1.3, -0.6, -0.6, 0.4],
            [-0.6, 0.4, -0.6, -0.6, -1.3, -1.3, -1.3, -0.6, -0.6, 0.4],
            4,
            1,
            inf,
            2,
            0.7213475204444815,
        ),
        (
            [1.04, 1.23, 0.92, 1.1, 1.34],
            [1.3, 1.1, 0.9, 1.2, 1.0],
            [1.0, 0.9, 1.2, 1.3, 1.1],
            3,
            1,
            inf,
            10,
            0.08685889638065034,
        ),
    ],
)
def test_ksg_cmi(data_x, data_y, cond, k, ksg_id, minkowski_p, base, expected):
    """Test the conditional
    Kraskov-Stoegbauer-Grassberger mutual information estimator."""
    est = KSGCMIEstimator(
        data_x,
        data_y,
        cond=cond,
        k=k,
        ksg_id=ksg_id,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,method,p_mi_1,p_mi_2,p_cmi_1,p_cmi_2",
    [
        (1, "permutation_test", 0.44, 0.42, 0.14, 0.2),
        (1, "bootstrap", 0.56, 0.38, 0.12, 0.1),
        (2, "permutation_test", 0.1, 0.1, 0.12, 0.12),
        (2, "bootstrap", 0.14, 0.1, 0.12, 0.12),
        (3, "permutation_test", 0.56, 0.54, 0.12, 0.12),
        (3, "bootstrap", 0.56, 0.48, 0.04, 0.04),
        (4, "permutation_test", 0.04, 0.04, 0.26, 0.26),
        (4, "bootstrap", 0.14, 0.06, 0.24, 0.16),
    ],
)
def test_ksg_mi_statistical_test(rng_int, method, p_mi_1, p_mi_2, p_cmi_1, p_cmi_2):
    """Test the KSG MI for p-values. Fix rng."""
    data_x, data_y, cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est_mi_1 = KSGMIEstimator(
        data_x, data_y, k=4, minkowski_p=inf, noise_level=0, base=2, seed=8, ksg_id=1
    )
    est_cmi_1 = KSGCMIEstimator(
        data_x,
        data_y,
        cond=cond,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base=2,
        seed=8,
        ksg_id=1,
    )
    test = est_mi_1.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_mi_1)
    test = est_cmi_1.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_cmi_1)

    est_mi_2 = KSGMIEstimator(
        data_x, data_y, k=4, minkowski_p=inf, noise_level=0, base=2, seed=8, ksg_id=2
    )
    est_cmi_2 = KSGCMIEstimator(
        data_x,
        data_y,
        cond=cond,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base=2,
        seed=8,
        ksg_id=2,
    )
    test = est_mi_2.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_mi_2)
    test = est_cmi_2.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_cmi_2)


def test_ksg_mi_variants_identical_data():
    """Test KSG MI variants with identical data to trigger ties."""
    # With identical data, Type I and Type II should behave differently
    # if noise is disabled.
    x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])

    # k=2. For any point, there are at least 2 other identical points (dist 0).
    # So the k-th neighbour in joint space is at distance 0.

    # Type I: dist < 0 is impossible. n_strict = 0.
    # Formula: psi(k) - (psi(1) + psi(1)) + psi(N)
    mi1 = mutual_information(x, y, approach="ksg", k=2, noise_level=0, ksg_id=1)

    # Type II: dist <= 0 includes all identical points.
    # Formula: psi(k) - 1/k - (psi(n_x) + psi(n_y)) + psi(N)
    mi2 = mutual_information(x, y, approach="ksg", k=2, noise_level=0, ksg_id=2)

    assert mi1 != mi2


@pytest.mark.parametrize("ksg_id", [1, 2])
def test_ksg_mi_with_noise_equivalence(ksg_id):
    """Test that both variants work correctly with noise."""
    np.random.seed(345)
    x = np.random.rand(100)
    y = x + np.random.normal(0, 0.1, 100)

    mi = mutual_information(x, y, approach="ksg", k=4, ksg_id=ksg_id)
    assert mi > 0


def test_ksg_invalid_id():
    """Test that invalid ksg_id raises ValueError."""
    x = np.random.rand(10)
    y = np.random.rand(10)

    # MI
    with pytest.raises(ValueError, match="ksg_id must be 1 or 2"):
        KSGMIEstimator(x, y, ksg_id=3)

    # CMI
    with pytest.raises(ValueError, match="ksg_id must be 1 or 2"):
        KSGCMIEstimator(x, y, cond=x, ksg_id=0)
