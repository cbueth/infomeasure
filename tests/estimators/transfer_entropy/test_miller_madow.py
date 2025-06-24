"""Tests for Miller-Madow transfer entropy estimators."""

import pytest
from numpy import e, log

import infomeasure as im
from infomeasure.estimators.transfer_entropy import (
    MillerMadowCTEEstimator,
    MillerMadowTEEstimator,
)
from tests.conftest import (
    discrete_random_variables,
    discrete_random_variables_condition,
    discrete_random_variables_shifted,
)


@pytest.mark.parametrize(
    "rng_int_prop,prop_time,expected_xy,expected_yx",
    [
        ((1, 0), -1, -0.0007855531188816489, 0.0009714028489564621),
        ((1, 0), 0, 1.0013524399779223, -0.007703638079202917),
        ((1, 0), 1, 0.0007713171212807551, 0.0033714765019765523),
        ((1, 0), 2, 0.005468273498077304, 0.005002851025326681),
        ((2, 0), 0, 1.0018860222012114, -0.006250717769893003),
        ((3, 0), 0, 0.9995962466838875, -0.0016141053971141378),
        ((4, 0), 0, 1.0006222189470164, -0.0013388958505127606),
        ((5, 0), 0, 1.0019801801536752, 0.007454844552658688),
        ((6, 0), 0, 1.0026711984332186, -0.0041355011752697395),
        ((1, 1), 0, -0.005474522329700256, -0.004157746524371422),
        ((1, 1), 1, 0.9964484343414046, 0.009110860436350756),
        ((1, 10), 10, 0.9954528957778193, -0.001593606420506006),
    ],
)
def test_miller_madow_autoregressive(rng_int_prop, prop_time, expected_xy, expected_yx):
    """Test the Miller-Madow transfer entropy estimator with autoregressive data."""
    data_source, data_dest = discrete_random_variables(
        rng_int_prop[0], prop_time=rng_int_prop[1]
    )
    est_xy = MillerMadowTEEstimator(data_source, data_dest, base=2, prop_time=prop_time)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_source, data_dest, approach="millermadow", base=2, prop_time=prop_time
    ) == pytest.approx(expected_xy)
    est_yx = MillerMadowTEEstimator(data_dest, data_source, base=2, prop_time=prop_time)
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)
    assert im.transfer_entropy(
        data_dest, data_source, approach="millermadow", base=2, prop_time=prop_time
    ) == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int,prop_time",
    [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
    ],
)
@pytest.mark.parametrize("high", [2, 4, 7])
@pytest.mark.parametrize("base", [2, e, 10])
def test_te_miller_madow_shifted(rng_int, prop_time, base, high):
    """Test the Miller-Madow transfer entropy estimator with shifted data."""
    source, dest, control = discrete_random_variables_shifted(
        rng_int, prop_time, low=0, high=high
    )
    est_xy = MillerMadowTEEstimator(source, dest, base=base, prop_time=prop_time - 1)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(log(high) / log(base), rel=0.02)
    est_xc = MillerMadowTEEstimator(source, control, base=base, prop_time=prop_time - 1)
    res_xc = est_xc.result()
    assert isinstance(res_xc, float)
    assert res_xc == pytest.approx(0.0, abs=0.018 * base * high)
    est_xy.local_vals()
    est_xc.local_vals()


@pytest.mark.parametrize(
    "rng_int,expected_xy,expected_yx",
    [
        (1, 1.004013980897449, -0.008419018106082823),
        (2, 1.002592166905885, -0.0015256831824253525),
        (3, 0.9969593033567905, -0.007092678468352244),
        (4, 1.0010645230000457, -0.0027721288421326123),
        (5, 1.0140921842688406, 0.001573335748625504),
        (6, 1.0091627180518317, -0.010348917207482555),
    ],
)
def test_cte_miller_madow_autoregressive(rng_int, expected_xy, expected_yx):
    """Test the conditional Miller-Madow transfer entropy estimator with
    autoregressive data."""
    data_source, data_dest, data_cond = discrete_random_variables_condition(rng_int)
    est_xy = MillerMadowCTEEstimator(data_source, data_dest, cond=data_cond, base=2)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_source,
        data_dest,
        cond=data_cond,
        approach="millermadow",
        base=2,
    ) == pytest.approx(expected_xy)
    est_yx = MillerMadowCTEEstimator(data_dest, data_source, cond=data_cond, base=2)
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)
    assert im.transfer_entropy(
        data_dest, data_source, cond=data_cond, approach="millermadow", base=2
    ) == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int_prop,step_size,src_hist_len,dest_hist_len,expected_xy,expected_yx",
    [
        ((1, 0), 1, 1, 1, 1.0013524399779223, -0.007703638079202917),
        ((1, 0), 2, 1, 1, -0.007437965157577657, 0.006599991106566637),
        ((1, 0), 1, 2, 1, 1.004130259677341, -0.00018154924248106608),
        ((1, 0), 1, 1, 2, 0.9993575043277158, 0.009269765066286496),
        ((1, 0), 2, 2, 2, 0.8127357616688071, 0.8216027274178114),
        ((2, 0), 1, 1, 1, 1.0018860222012114, -0.006250717769893003),
        ((2, 0), 2, 1, 1, 0.009385229115448278, -0.010980300312909468),
        ((2, 0), 1, 2, 1, 1.0004454763595039, -0.0034590896831303564),
        ((2, 0), 1, 1, 2, 0.9976050523368977, -0.006341475878445267),
        ((2, 0), 2, 2, 2, 0.8377250837470137, 0.8116290471779718),
    ],
)
def test_miller_madow_te_slicing(rng_int_prop, step_size, src_hist_len, dest_hist_len, expected_xy, expected_yx):
    """Test the Miller-Madow transfer entropy estimator with different slicing parameters."""
    data_source, data_dest = discrete_random_variables(
        rng_int_prop[0], prop_time=rng_int_prop[1]
    )
    est_xy = MillerMadowTEEstimator(
        data_source,
        data_dest,
        base=2,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
    )
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    est_yx = MillerMadowTEEstimator(
        data_dest,
        data_source,
        base=2,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
    )
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int,base,expected_xy,expected_yx",
    [
        (1, 2, 1.0013524399779223, -0.007703638079202917),
        (1, e, 0.6940846205175191, -0.005339755014653681),
        (1, 10, 0.301437120664671, -0.00231902613757935),
        (2, 2, 1.0018860222012114, -0.006250717769893003),
        (2, e, 0.6944544715311883, -0.004332667398677312),
        (2, 10, 0.30159774491903385, -0.001881653543167704),
    ],
)
def test_miller_madow_te_base(rng_int, base, expected_xy, expected_yx):
    """Test the Miller-Madow transfer entropy estimator with different bases."""
    data_source, data_dest = discrete_random_variables(rng_int)
    est_xy = MillerMadowTEEstimator(data_source, data_dest, base=base)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    est_yx = MillerMadowTEEstimator(data_dest, data_source, base=base)
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int,base,expected_xy,expected_yx",
    [
        (1, 2, 1.004013980897449, -0.008419018106082823),
        (1, e, 0.6959294601018337, -0.005835618663314349),
        (1, 10, 0.30223832431613556, -0.0025343769839690604),
        (2, 2, 1.002592166905885, -0.0015256831824253525),
        (2, e, 0.6949439337423003, -0.0010575229963258187),
        (2, 10, 0.30181031565642, -0.0004592764017901235),
    ],
)
def test_miller_madow_cte_base(rng_int, base, expected_xy, expected_yx):
    """Test the conditional Miller-Madow transfer entropy estimator with different bases."""
    data_source, data_dest, data_cond = discrete_random_variables_condition(rng_int)
    est_xy = MillerMadowCTEEstimator(data_source, data_dest, cond=data_cond, base=base)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    est_yx = MillerMadowCTEEstimator(data_dest, data_source, cond=data_cond, base=base)
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)


def test_miller_madow_te_uncoupled(default_rng):
    """Test the Miller-Madow transfer entropy estimator with uncoupled data."""
    data_source = default_rng.integers(0, 4, 1000)
    data_dest = default_rng.integers(0, 4, 1000)
    est_xy = MillerMadowTEEstimator(data_source, data_dest, base=2)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(0.0, abs=0.1)
