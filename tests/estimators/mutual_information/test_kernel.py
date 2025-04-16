"""Explicit kernel mutual information estimator tests."""

import pytest
from numpy import ndarray, equal, allclose

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.estimators.mutual_information import (
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
    assert isinstance(est.result(), float)
    assert isinstance(est.local_vals(), ndarray)


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
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


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
    data_x, data_y, cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KernelCMIEstimator(
        data_x, data_y, cond=cond, bandwidth=bandwidth, kernel=kernel, base=2
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize("rng_int", [1, 2])
@pytest.mark.parametrize("workers", [-1, 1])
@pytest.mark.parametrize("kernel", ["gaussian", "box"])
def test_kernel_mi_parallelization(rng_int, workers, kernel):
    """Test the Kernel MI estimator with different worker counts."""
    data_x, data_y = generate_autoregressive_series(
        rng_int, 0.5, 0.6, 0.4, length=int(20001)
    )
    est_parallel = KernelMIEstimator(
        data_x,
        data_y,
        bandwidth=0.5,
        kernel=kernel,
        base=2,
        workers=workers,
    )
    est_serial = KernelMIEstimator(
        data_x,
        data_y,
        cond=None,
        bandwidth=0.5,
        kernel=kernel,
        base=2,
        workers=1,
    )
    assert est_parallel.global_val() == pytest.approx(est_serial.global_val())
    assert allclose(est_parallel.local_vals(), est_serial.local_vals())
