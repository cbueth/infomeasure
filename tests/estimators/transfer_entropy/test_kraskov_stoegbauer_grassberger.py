"""
Explicit tests for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator.
"""

import pytest
from numpy import ndarray, inf

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.estimators.transfer_entropy import KSGTEEstimator, KSGCTEEstimator


@pytest.mark.parametrize(
    "rng_int,k,minkowski_p,base,expected",
    [
        (5, 4, 2, "e", -0.1464972645512768),
        (5, 4, 3, "e", 0.20521970071775472),
        (5, 4, inf, "e", 0.5959660770508142),
        (5, 16, 2, "e", -0.09411416419535731),
        (5, 16, 3, "e", 0.23892165815373245),
        (5, 16, inf, "e", 0.6034525014935584),
        (6, 4, 2, "e", -0.14070180426292878),
        (6, 4, 3, "e", 0.20704153706669481),
        (6, 4, inf, "e", 0.5660288107639446),
        (7, 4, 2, "e", -0.1333354231111923),
        (7, 4, 3, "e", 0.21280380426208417),
        (7, 4, inf, "e", 0.598763591023636),
        (7, 4, 2, 2.0, -0.19236235),
        (7, 4, 3, 10, 0.092419517),
        (7, 4, inf, 5, 0.372032736),
    ],
)
def test_ksg_te(rng_int, k, minkowski_p, base, expected):
    """Test the KSG transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KSGTEEstimator(
        data_source,
        data_dest,
        k=k,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,expected",
    [
        (5, 1, 1, 1, 1, 0.4979940181649),
        (6, 1, 1, 1, 1, 0.498845843022),
        (7, 1, 1, 1, 1, 0.513635125765),
        (5, 0, 1, 1, 1, 0.595966077050),
        (5, 1, 2, 1, 1, 0.51980907955),
        (5, 1, 3, 1, 1, 0.513413348695),
        (5, 1, 1, 2, 1, 0.914015718313),
        (5, 1, 1, 1, 2, 1.113892408337),
        (5, 1, 1, 2, 2, 1.580475709527),
    ],
)
def test_ksg_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
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
        noise_level=0,  # for reproducibility
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,k,minkowski_p,base,expected",
    [
        (5, 4, 2, "e", -0.14824377373),
        (5, 4, 3, "e", 0.492061479471),
        (5, 4, inf, "e", 1.1736338791),
        (5, 16, 2, "e", -0.0629498022),
        (5, 16, 3, "e", 0.5019661670),
        (5, 16, inf, "e", 1.1344990340),
        (6, 4, 2, "e", -0.1120271189),
        (6, 4, 3, "e", 0.5267832946),
        (6, 4, inf, "e", 1.2010598113),
        (7, 4, 2, "e", -0.17448870909),
        (7, 4, 3, "e", 0.4594946023),
        (7, 4, inf, "e", 1.1573234261),
        (7, 4, 2, 2.0, -0.25173399),
        (7, 4, 3, 10, 0.199555970),
        (7, 4, inf, 5.0, 0.7190854),
    ],
)
def test_ksg_cte(rng_int, k, minkowski_p, base, expected):
    """Test the conditional KSG transfer entropy estimator."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        k=k,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,step_size,src_hist_len,dest_hist_len,cond_hist_len,expected",
    [
        (5, 1, 1, 1, 1, 1.17363387),
        (6, 1, 1, 1, 1, 1.20105981),
        (7, 1, 1, 1, 1, 1.15732342),
        (5, 1, 1, 1, 1, 1.17363387),
        (5, 2, 1, 1, 1, 1.11722357),
        (5, 3, 1, 1, 1, 1.11096789),
        (5, 1, 2, 1, 1, 1.65521713),
        (5, 1, 1, 2, 1, 1.75221232),
        (5, 1, 1, 1, 2, 1.72224612),
        (5, 1, 2, 2, 1, 2.17620552),
        (5, 1, 2, 2, 2, 2.58013058),
    ],
)
def test_ksg_cte_slicing(
    rng_int,
    step_size,
    src_hist_len,
    dest_hist_len,
    cond_hist_len,
    expected,
):
    """Test the conditional KSG transfer entropy estimator with slicing."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        noise_level=0,  # for reproducibility
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)
