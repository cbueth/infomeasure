"""Explicit discrete transfer entropy estimator tests."""

import pytest
from numpy import e, log

import infomeasure as im
from infomeasure.estimators.transfer_entropy import (
    DiscreteCTEEstimator,
    DiscreteTEEstimator,
)
from tests.conftest import (
    discrete_random_variables,
    discrete_random_variables_condition,
    discrete_random_variables_shifted,
)


@pytest.mark.parametrize(
    "rng_int_prop,prop_time,expected_xy,expected_yx",
    [
        ((1, 0), -1, 0.025234998, 0.00964492),
        ((1, 0), 0, 1.004240718338, 0.01829086716204),
        ((1, 0), 1, 0.0094448344, 0.029392028),
        ((1, 0), 2, 0.03151492418, 0.031049501713),
        ((2, 0), 0, 1.00477430056, 0.0197437874713),
        ((3, 0), 0, 1.00248452504, 0.02438039984412),
        ((4, 0), 0, 1.0035104973, 0.0246556093),
        ((5, 0), 0, 1.0048684585, 0.0334493497),
        ((6, 0), 0, 1.00555947, 0.0218590040),
        ((1, 1), 0, 0.020519982911, 0.02183675871687),
        ((1, 1), 1, 0.999339606768, 0.035131412276),
        ((1, 10), 10, 0.99837037816, 0.024663735071),
    ],
)
def test_discrete_autoregressive(rng_int_prop, prop_time, expected_xy, expected_yx):
    """Test the discrete transfer entropy estimator with autoregressive data."""
    data_source, data_dest = discrete_random_variables(
        rng_int_prop[0], prop_time=rng_int_prop[1]
    )
    est_xy = DiscreteTEEstimator(data_source, data_dest, base=2, prop_time=prop_time)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_source, data_dest, approach="discrete", base=2, prop_time=prop_time
    ) == pytest.approx(expected_xy)
    est_yx = DiscreteTEEstimator(data_dest, data_source, base=2, prop_time=prop_time)
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)
    assert im.transfer_entropy(
        data_dest, data_source, approach="discrete", base=2, prop_time=prop_time
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
def test_te_discrete_shifted(rng_int, prop_time, base, high):
    """Test the discrete transfer entropy estimator with shifted data."""
    source, dest, control = discrete_random_variables_shifted(
        rng_int, prop_time, low=0, high=high
    )
    est_xy = DiscreteTEEstimator(source, dest, base=base, prop_time=prop_time - 1)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(log(high) / log(base), rel=0.02)
    est_xc = DiscreteTEEstimator(source, control, base=base, prop_time=prop_time - 1)
    res_xc = est_xc.result()
    assert isinstance(res_xc, float)
    assert res_xc == pytest.approx(0.0, abs=0.018 * base * high)
    est_xy.local_vals()
    est_xc.local_vals()


@pytest.mark.parametrize(
    "rng_int,expected_xy,expected_yx",
    [
        (1, 1.00979053, 0.02624032),
        (2, 1.00836872, 0.03313365),
        (3, 1.00273586, 0.02756666),
        (4, 1.00684107, 0.03188721),
        (5, 1.01986874, 0.03623267),
        (6, 1.01493927, 0.02431042),
    ],
)
def test_cte_discrete_autoregressive(rng_int, expected_xy, expected_yx):
    """Test the conditional discrete transfer entropy estimator with
    autoregressive data."""
    data_source, data_dest, data_cond = discrete_random_variables_condition(rng_int)
    est_xy = DiscreteCTEEstimator(data_source, data_dest, cond=data_cond, base=2)
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_source,
        data_dest,
        cond=data_cond,
        approach="discrete",
        base=2,
    ) == pytest.approx(expected_xy)
    est_yx = DiscreteCTEEstimator(data_dest, data_source, cond=data_cond, base=2)
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)
    assert im.transfer_entropy(
        data_dest, data_source, cond=data_cond, approach="discrete", base=2
    ) == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int_prop,step_size,src_hist_len,dest_hist_len,expected_xy,expected_yx",
    [
        ((1, 0), 1, 1, 1, 1.004240718338, 0.01829086716204),
        ((1, 0), 2, 1, 1, 0.044603138521, 0.05864109478),
        ((1, 0), 1, 2, 1, 1.018586121810, 0.043963834185),
        ((1, 0), 1, 1, 2, 1.01092219403, 0.128475623742),
        ((1, 1), 1, 1, 1, 0.020519982, 0.02183675871),
        ((1, 2), 1, 1, 1, 0.030291914, 0.025516281045),
    ],
)
def test_discrete_te_slicing(
    rng_int_prop, step_size, src_hist_len, dest_hist_len, expected_xy, expected_yx
):
    """Test the discrete transfer entropy estimator with slicing."""
    data_source, data_dest = discrete_random_variables(
        rng_int_prop[0], prop_time=rng_int_prop[1]
    )
    assert im.transfer_entropy(
        data_source,
        data_dest,
        approach="discrete",
        base=2,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
    ) == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_dest,
        data_source,
        approach="discrete",
        base=2,
        step_size=step_size,
        src_hist_len=dest_hist_len,
        dest_hist_len=src_hist_len,
    ) == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int,base,expected_xy,expected_yx",
    [
        (1, 2, 1.004240718, 0.01829086716204),
        (1, 10, 0.302306579086, 0.00550609966247),
        (1, 16, 0.251060179584, 0.00457271679051),
        (2, 2, 1.00477430056, 0.0197437874713),
        (2, 10, 0.302467203341, 0.00594347225689),
        (2, 16, 0.251193575140, 0.00493594686783),
    ],
)
def test_discrete_te_base(rng_int, base, expected_xy, expected_yx):
    """Test the discrete transfer entropy estimator with different bases."""
    data_source, data_dest = discrete_random_variables(rng_int)
    assert im.transfer_entropy(
        data_source, data_dest, approach="discrete", base=base
    ) == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_dest, data_source, approach="discrete", base=base
    ) == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "rng_int,base,expected_xy,expected_yx",
    [
        (1, 2, 1.00979053, 0.026240322),
        (1, "e", 0.69993346, 0.018188405),
        (1, 10, 0.30397724, 0.007899124),
        (1, 16, 0.25244763, 0.006560080),
        (2, 2, 1.00836872, 0.033133657),
        (2, "e", 0.69894793, 0.022966501),
        (2, 10, 0.30354923, 0.009974224),
        (2, 16, 0.25209218, 0.008283414),
    ],
)
def test_discrete_cte_base(rng_int, base, expected_xy, expected_yx):
    """Test the conditional discrete transfer entropy estimator with different bases."""
    data_source, data_dest, data_cond = discrete_random_variables_condition(rng_int)
    assert im.transfer_entropy(
        data_source, data_dest, cond=data_cond, approach="discrete", base=base
    ) == pytest.approx(expected_xy)
    assert im.transfer_entropy(
        data_dest, data_source, cond=data_cond, approach="discrete", base=base
    ) == pytest.approx(expected_yx)


def test_discrete_te_uncoupled(default_rng):
    """Test the discrete transfer entropy estimator with uncoupled data."""
    x = default_rng.integers(0, 4, 10000)
    y = default_rng.integers(0, 4, 10000)
    assert im.transfer_entropy(x, y, approach="discrete") == pytest.approx(
        0.0, abs=1e-2
    )
    assert im.transfer_entropy(y, x, approach="discrete") == pytest.approx(
        0.0, abs=1e-2
    )
