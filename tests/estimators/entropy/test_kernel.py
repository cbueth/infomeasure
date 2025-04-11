"""Explicit Kernel entropy estimator tests."""

import pytest
from numpy import allclose, ndarray

from infomeasure.estimators.entropy import KernelEntropyEstimator
from tests.conftest import generate_autoregressive_series


@pytest.mark.parametrize("bandwidth", [0.1, 1, 10])
@pytest.mark.parametrize("kernel", ["box", "gaussian"])
def test_kernel_entropy(bandwidth, kernel, default_rng):
    """Test the Kernel entropy estimator by design."""
    data = default_rng.normal(0, 10, 100)
    est = KernelEntropyEstimator(data, bandwidth=bandwidth, kernel=kernel)
    assert 0 <= est.global_val()


@pytest.mark.parametrize(
    "data,bandwidth,kernel,expected",
    [
        ([1, 0, 1, 1, 1, 4, 23, 6, 1], 1, "gaussian", 4.75097076318),
        ([1, 0, 1, 1, 1, 4, 23, 6, 1], 1, "box", 1.879964948727),
        ([1, 0, 1, 0, 1, 0], 1, "gaussian", 1.207667500053),
        ([1, 0, 1, 0, 1, 0], 1, "box", 1.0),
        ([1, 2, 3, 4, 5], 1, "gaussian", 2.740893667603),
        ([1, 2, 3, 4, 5], 1, "box", 2.32192809488),
        ([1, 2, 3, 4, 5], 2, "gaussian", 3.243552771805),
        ([1, 2, 3, 4, 5], 2, "box", 1.970950594454),
        (
            [
                [1, 4, 2],
                [3, 2, 1],
                [1, 2, 3],
                [2, 3, 1],
                [3, 1, 2],
                [2, 1, 3],
                [1, 3, 2],
                [2, 3, 1],
                [10, 20, 30],
                [30, 20, 10],
                [-7, 5, 1],
            ],
            4,
            "gaussian",
            17.6074091260,
        ),
        (
            [
                [1, 4, 2],
                [3, 2, 1],
                [1, 2, 3],
                [2, 3, 1],
                [3, 1, 2],
                [2, 1, 3],
                [1, 3, 2],
                [2, 3, 1],
                [10, 20, 30],
                [30, 20, 10],
                [-7, 5, 1],
            ],
            4,
            "box",
            7.35037049637,
        ),
    ],
)
def test_kernel_entropy_explicit(data, bandwidth, kernel, expected):
    """Test the Kernel entropy estimator with specific values."""
    est = KernelEntropyEstimator(data, bandwidth=bandwidth, kernel=kernel, base=2)
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize("bandwidth", [0, -1, -10, None])
@pytest.mark.parametrize("kernel", ["box", "gaussian"])
def test_kernel_entropy_invalid_bandwidth(bandwidth, kernel):
    """Test the Kernel entropy estimator with invalid bandwidth."""
    data = list(range(1000))
    with pytest.raises(ValueError, match="The bandwidth must be a positive number."):
        est = KernelEntropyEstimator(data, bandwidth=bandwidth, kernel=kernel)
        est.result()


@pytest.mark.parametrize("kernel", ["invalid_kernel", None, 1])
def test_kernel_entropy_invalid_kernel(kernel):
    """Test the Kernel entropy estimator with invalid kernel."""
    data = list(range(1000))
    with pytest.raises(
        ValueError, match=f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'."
    ):
        est = KernelEntropyEstimator(data, kernel=kernel, bandwidth=1)
        est.result()


@pytest.mark.parametrize("rng_int", [1, 2])
@pytest.mark.parametrize("workers", [-1, 1])
@pytest.mark.parametrize("kernel", ["gaussian", "box"])
def test_kernel_mi_parallelization(rng_int, workers, kernel):
    """Test the Kernel MI estimator with different worker counts."""
    data_x, _ = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4, length=20001)
    est_parallel = KernelEntropyEstimator(
        data_x,
        bandwidth=0.5,
        kernel=kernel,
        base=2,
        workers=workers,
    )
    est_serial = KernelEntropyEstimator(
        data_x,
        bandwidth=0.5,
        kernel=kernel,
        base=2,
        workers=1,
    )
    assert est_parallel.global_val() == pytest.approx(est_serial.global_val())
    assert allclose(est_parallel.local_vals(), est_serial.local_vals())
