"""Explicit kernel mutual information estimator tests."""

import pytest
from numpy import ndarray, std

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.measures.mutual_information import (
    KernelMIEstimator,
    KernelCMIEstimator,
)

KERNELS = ["gaussian", "box"]


@pytest.mark.parametrize("bandwidth", [0.1, 1, 10])
@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_mi(bandwidth, kernel, default_rng):
    """Test the kernel mutual information estimator."""
    data_x = default_rng.normal(0, 1, 100)
    data_y = default_rng.normal(0, 1, 100)
    est = KernelMIEstimator(data_x, data_y, bandwidth=bandwidth, kernel=kernel)
    res = est.results()
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert isinstance(res[1], ndarray)
    assert isinstance(res[2], float)


@pytest.mark.parametrize(
    "rng_int,bandwidth,kernel,expected",
    [
        (5, 0.01, "gaussian", 4.469608036963662),
        (5, 0.01, "box", 9.538895559637568),
        (5, 0.1, "gaussian", 0.4206212233457012),
        (5, 0.1, "box", 7.067571502526608),
        (5, 1, "gaussian", 0.04866184805475981),
        (5, 1, "box", 1.8066997932708635),
    ],
)
def test_kernel_mi_values(rng_int, bandwidth, kernel, expected):
    """Test the kernel mutual information estimator with specific values."""
    data_x, data_y = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KernelMIEstimator(data_x, data_y, bandwidth=bandwidth, kernel=kernel, base=2)
    assert est.results()[0] == pytest.approx(expected)
    assert est.results()[2] == std(est.results()[1])


@pytest.mark.parametrize(
    "rng_int,bandwidth,kernel,expected",
    [
        (5, 0.01, "gaussian", 2.7659996851),
        (5, 0.01, "box", 0.2563985),
        (5, 0.1, "gaussian", 1.76267892),
        (5, 0.1, "box", 1.7303813472),
        (5, 1, "gaussian", 0.0592152641),
        (5, 1, "box", 3.6203671293),
    ],
)
def test_kernel_cmi_values(rng_int, bandwidth, kernel, expected):
    """Test the kernel conditional mutual information estimator with specific values."""
    data_x, data_y, data_z = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KernelCMIEstimator(
        data_x, data_y, data_z, bandwidth=bandwidth, kernel=kernel, base=2
    )
    assert est.results()[0] == pytest.approx(expected)
    assert est.results()[2] == std(est.results()[1])
