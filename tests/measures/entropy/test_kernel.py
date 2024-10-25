"""Explicit Kernel entropy estimator tests."""

import pytest

from infomeasure.measures.entropy import KernelEntropyEstimator


@pytest.mark.parametrize("bandwidth", [0.1, 1, 10])
@pytest.mark.parametrize("kernel", ["box", "gaussian"])
def test_kernel_entropy(bandwidth, kernel, default_rng):
    """Test the Kernel entropy estimator by design."""
    data = default_rng.normal(0, 10, 100)
    est = KernelEntropyEstimator(data, bandwidth=bandwidth, kernel=kernel)
    assert 0 <= est.results()


@pytest.mark.parametrize("bandwidth", [0, -1, -10, None])
@pytest.mark.parametrize("kernel", ["box", "gaussian"])
def test_kernel_entropy_invalid_bandwidth(bandwidth, kernel):
    """Test the Kernel entropy estimator with invalid bandwidth."""
    data = list(range(1000))
    with pytest.raises(ValueError, match="The bandwidth must be a positive number."):
        est = KernelEntropyEstimator(data, bandwidth=bandwidth, kernel=kernel)
        est.results()


@pytest.mark.parametrize("kernel", ["invalid_kernel", None, 1])
def test_kernel_entropy_invalid_kernel(kernel):
    """Test the Kernel entropy estimator with invalid kernel."""
    data = list(range(1000))
    with pytest.raises(
        ValueError, match=f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'."
    ):
        est = KernelEntropyEstimator(data, kernel=kernel, bandwidth=1)
        est.results()
