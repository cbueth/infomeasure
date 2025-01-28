"""Explicit tests for the Renyi transfer entropy estimator."""

import pytest
from numpy import ndarray, std

from tests.conftest import generate_autoregressive_series
from infomeasure.measures.transfer_entropy import RenyiTEEstimator


@pytest.mark.parametrize(
    "k,alpha,expected",
    [
        ([2, 1.0, 0.1328256688984]),
        ([2, 1.1, 0.1225837946434]),
        ([3, 2.0, -0.1227381271547]),
        ([1, 1.1, -0.03010890843113]),
        ([2, 2.0, 0.0449216611089]),
        ([4, 1.0, 0.06046248740893]),
    ],
)
def test_renyi_te(k, alpha, expected):
    """Test the Renyi transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(5, 0.5, 0.6, 0.4)
    est = RenyiTEEstimator(
        data_source,
        data_dest,
        k=k,
        alpha=alpha,
        noise_level=0,  # for reproducibility
        base=2,
    )
    res = est.results()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,base,k,alpha,expected",
    [
        (5, 0, 1, 1, 1, 2.0, 2, 1.0, 0.1328256688984),
        (5, 0, 1, 1, 1, 2.0, 4, 1.0, 0.06046248740893),
        (5, 1, 1, 1, 1, 2.0, 4, 1.0, -0.00097675392276),
        (6, 0, 1, 1, 1, 2.0, 2, 1.0, 0.0978424366616),
        (5, 1, 2, 1, 1, 2.0, 2, 1.0, 0.02631960477116),
        (5, 1, 3, 1, 1, 2.0, 2, 1.0, -0.0707019638402),
        (5, 1, 1, 2, 1, 2.0, 2, 1.0, -0.03704319399725),
        (5, 1, 1, 1, 2, 2.0, 2, 1.0, -0.0350563983761),
        (5, 1, 1, 2, 2, 2.0, 2, 1.0, -0.0517794607390),
    ],
)
def test_renyi_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    k,
    alpha,
    expected,
):
    """Test the Renyi transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = RenyiTEEstimator(
        data_source,
        data_dest,
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        k=k,
        alpha=alpha,
        noise_level=0,  # for reproducibility
    )
    res = est.results()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
