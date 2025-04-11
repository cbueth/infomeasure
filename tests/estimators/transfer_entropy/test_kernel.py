"""Explicit tests for transfer entropy kernel functions."""

import pytest
from numpy import ndarray, allclose

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.estimators.transfer_entropy import (
    KernelTEEstimator,
    KernelCTEEstimator,
)


@pytest.mark.parametrize(
    "rng_int,bandwidth,kernel,expected",
    [
        (5, 0.01, "gaussian", 2.844102091538639),
        (5, 0.01, "box", 0.1712660035085565),
        (6, 0.1, "gaussian", 1.806522663943940),
        (6, 0.1, "box", 1.283448741027693),
        (7, 1, "gaussian", 0.1769960416964969),
        (7, 1, "box", 3.234507644104068),
    ],
)
def test_kernel_te(rng_int, bandwidth, kernel, expected):
    """Test the kernel transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KernelTEEstimator(
        data_source,
        data_dest,
        bandwidth=bandwidth,
        kernel=kernel,
        base=2,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,base,expected",
    [
        (5, 1, 1, 1, 1, 2.0, 0.1694336047144764),
        (5, 1, 1, 1, 1, 10.0, 0.051004597292531526),
        (6, 1, 1, 1, 1, 2.0, 0.19851131013396928),
        (7, 1, 1, 1, 1, 2.0, 0.1721940130332775),
        (5, 0, 1, 1, 1, 2.0, 0.1712660035085565),
        (5, 1, 2, 1, 1, 2.0, 0.10040160642570281),
        (5, 1, 3, 1, 1, 2.0, 0.04694265813470213),
        (5, 1, 1, 2, 1, 2.0, 0.17160956620365966),
        (5, 1, 1, 1, 2, 2.0, 9.61027951e-16),
        (5, 1, 1, 2, 2, 2.0, 0.0),
    ],
)
def test_kernel_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    expected,
):
    """Test the kernel transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KernelTEEstimator(
        data_source,
        data_dest,
        bandwidth=0.01,
        kernel="box",
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,bandwidth,kernel,expected",
    [
        (5, 0.01, "gaussian", 0.153130431),
        (5, 0.01, "box", 3.2034265e-16),
        (6, 0.1, "gaussian", 1.7457075600),
        (6, 0.1, "box", 0.00200200200200),
        (7, 1, "gaussian", 0.13330861957),
        (7, 1, "box", 0.396311098),
    ],
)
def test_kernel_cte(rng_int, bandwidth, kernel, expected):
    """Test the conditional kernel transfer entropy estimator."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KernelCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        bandwidth=bandwidth,
        kernel=kernel,
        base=2,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,step_size,src_hist_len,dest_hist_len,cond_hist_len,base,expected",
    [
        (5, 1, 1, 1, 1, 2.0, 3.2034e-16),
        (5, 1, 1, 1, 1, 10.0, 9.6433e-17),
        (6, 1, 1, 1, 1, 2.0, 3.2034e-16),
        (7, 1, 1, 1, 1, 2.0, 3.2034e-16),
        (5, 2, 1, 1, 1, 2.0, 6.4069e-16),
        (5, 3, 1, 1, 1, 2.0, 3.2034e-16),
        (5, 1, 2, 1, 1, 2.0, 0.0),
        (5, 1, 1, 2, 1, 2.0, -4.8051e-16),
        (5, 1, 1, 1, 2, 2.0, -4.8051e-16),
        (5, 1, 2, 2, 1, 2.0, -3.2034e-16),
        (5, 1, 2, 2, 2, 2.0, 6.4069e-16),
        # TODO: Add test with larger expected value
    ],
)
def test_kernel_cte_slicing(
    rng_int,
    step_size,
    src_hist_len,
    dest_hist_len,
    cond_hist_len,
    base,
    expected,
):
    """Test the conditionalkernel transfer entropy estimator with slicing."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KernelCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        bandwidth=0.01,
        kernel="box",
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize("rng_int", [1, 2])
@pytest.mark.parametrize("workers", [-1, 1])
@pytest.mark.parametrize("kernel", ["gaussian", "box"])
def test_kernel_te_parallelization(rng_int, workers, kernel):
    """Test the Kernel TE estimator with different worker counts."""
    data_x, data_y = generate_autoregressive_series(
        rng_int, 0.5, 0.6, 0.4, length=int(20001)
    )
    est_parallel = KernelTEEstimator(
        data_x,
        data_y,
        bandwidth=0.5,
        kernel=kernel,
        base=2,
        workers=workers,
    )
    est_parallel.result()
    est_serial = KernelTEEstimator(
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
