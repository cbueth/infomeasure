"""Explicit kernel mutual information estimator tests."""

import pytest
from numpy import ndarray

from infomeasure.measures.mutual_information import KernelMIEstimator

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
