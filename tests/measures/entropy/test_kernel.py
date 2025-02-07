"""Explicit Kernel entropy estimator tests."""

import pytest

from infomeasure.estimators.entropy import KernelEntropyEstimator


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
    res = est.results()
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert res[0] == pytest.approx(expected)
    assert res[2] == pytest.approx(res[1].std())


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
