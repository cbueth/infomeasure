"""Explicit tests for transfer entropy kernel functions."""

import pytest
from numpy import ndarray, std

from tests.conftest import generate_autoregressive_series
from infomeasure.measures.transfer_entropy import KernelTEEstimator


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
    )
    res = est.results()
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert isinstance(res[1], ndarray)
    assert isinstance(res[2], float)
    assert res[0] == pytest.approx(expected)
    assert res[2] == pytest.approx(std(res[1]))


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
    res = est.results()
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert isinstance(res[1], ndarray)
    assert isinstance(res[2], float)
    assert res[0] == pytest.approx(expected)
    assert res[2] == pytest.approx(std(res[1]))
