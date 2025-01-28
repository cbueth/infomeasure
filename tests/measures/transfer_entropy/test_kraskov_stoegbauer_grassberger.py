"""
Explicit tests for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator.
"""

import pytest
from numpy import ndarray, std, inf

from tests.conftest import generate_autoregressive_series
from infomeasure.measures.transfer_entropy import KSGTEEstimator


@pytest.mark.parametrize(
    "rng_int,k,minkowski_p,expected",
    [
        (5, 4, 2, -0.1464972645512768),
        (5, 4, 3, 0.20521970071775472),
        (5, 4, inf, 0.5959660770508142),
        (5, 16, 2, -0.09411416419535731),
        (5, 16, 3, 0.23892165815373245),
        (5, 16, inf, 0.6034525014935584),
        (6, 4, 2, -0.14070180426292878),
        (6, 4, 3, 0.20704153706669481),
        (6, 4, inf, 0.5660288107639446),
        (7, 4, 2, -0.1333354231111923),
        (7, 4, 3, 0.21280380426208417),
        (7, 4, inf, 0.598763591023636),
    ],
)
def test_ksg_te(rng_int, k, minkowski_p, expected):
    """Test the KSG transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KSGTEEstimator(
        data_source,
        data_dest,
        k=k,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=2,
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
        (5, 1, 1, 1, 1, 2.0, 0.4979940181649),
        (5, 1, 1, 1, 1, 10.0, 0.4979940181649),
        (6, 1, 1, 1, 1, 2.0, 0.498845843022),
        (7, 1, 1, 1, 1, 2.0, 0.513635125765),
        (5, 0, 1, 1, 1, 2.0, 0.595966077050),
        (5, 1, 2, 1, 1, 2.0, 0.51980907955),
        (5, 1, 3, 1, 1, 2.0, 0.513413348695),
        (5, 1, 1, 2, 1, 2.0, 0.914015718313),
        (5, 1, 1, 1, 2, 2.0, 1.113892408337),
        (5, 1, 1, 2, 2, 2.0, 1.580475709527),
    ],
)
def test_ksg_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    expected,
):
    """Test the KSG transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KSGTEEstimator(
        data_source,
        data_dest,
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        noise_level=0,  # for reproducibility
    )
    res = est.results()
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert isinstance(res[1], ndarray)
    assert isinstance(res[2], float)
    assert res[0] == pytest.approx(expected)
    assert res[2] == pytest.approx(std(res[1]))
