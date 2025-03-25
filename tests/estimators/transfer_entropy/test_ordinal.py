"""Explicit ordinal / permutation transfer entropy estimator."""

import pytest
from numpy import isnan

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
    discrete_random_variables_condition,
)
from infomeasure.estimators.transfer_entropy import (
    OrdinalTEEstimator,
    OrdinalCTEEstimator,
)


@pytest.mark.parametrize("data_len", [1, 2, 3, 10, 100])
@pytest.mark.parametrize("embedding_dim", [2, 3, 5])
@pytest.mark.parametrize("step_size", [1, 2, 3])
@pytest.mark.parametrize("prop_time", [0, 1, 4])
def test_ordinal_te(data_len, embedding_dim, step_size, prop_time, default_rng):
    """Test the discrete transfer entropy estimator."""
    source = default_rng.integers(0, 10, data_len)
    dest = default_rng.integers(0, 10, data_len)
    if data_len - abs(prop_time * step_size) <= (embedding_dim - 1) * step_size + 1:
        with pytest.raises(ValueError):
            est = OrdinalTEEstimator(
                source,
                dest,
                embedding_dim=embedding_dim,
                step_size=step_size,
                prop_time=prop_time,
            )
            est.result()
        return
    if embedding_dim == 1:
        est = OrdinalTEEstimator(
            source,
            dest,
            embedding_dim,
            step_size=step_size,
            prop_time=prop_time,
        )
        assert est.global_val() == 0
        for i in est.local_val():
            assert i == 0
        if len(est.local_val()) > 0:
            assert est.std_val() == 0
        else:
            assert isnan(est.std_val())
        return
    est = OrdinalTEEstimator(
        source,
        dest,
        embedding_dim,
        step_size=step_size,
        prop_time=prop_time,
    )
    max_val = est._log_base(data_len)
    assert 0 <= est.global_val() <= max_val


@pytest.mark.parametrize("embedding_dim", [-1, 1.0, "a", 1.5, 2.0])
def test_ordinal_te_invalid_embedding_dim(embedding_dim, default_rng):
    """Test the discrete transfer entropy estimator with invalid embedding_dim."""
    data = list(range(10))
    with pytest.raises(ValueError):
        OrdinalTEEstimator(data, data, embedding_dim=embedding_dim)


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected",
    [
        (5, 1, 0.0),
        (6, 1, 0.0),
        (5, 2, 0.003214612864590),
        (6, 2, 0.00224052033287),
        (7, 2, 0.00433834421171),
        (5, 3, 0.0010093514626),
        (6, 3, 0.000786386118684),
        (5, 4, 0.001311055546311),
        (6, 4, 0.001369174129191),
        (5, 5, 0.001663653489971),
        (6, 5, 0.001694007216101),
    ],
)
def test_ordinal_te(rng_int, embedding_dim, expected):
    """Test the ordinal transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = OrdinalTEEstimator(
        data_source, data_dest, embedding_dim=embedding_dim, base=2, stable=True
    )
    res = est.result()
    if embedding_dim == 1:
        assert isinstance(res, float)
        assert res == 0.0
        return
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,base,embedding_dim,expected",
    [
        (5, 0, 1, 1, 1, 2.0, 2, 0.003214612864590),
        (5, 1, 2, 1, 1, 2.0, 2, 0.000577684847751),
        (5, 1, 3, 1, 1, 2.0, 2, 0.000865495306628),
        (5, 1, 1, 2, 1, 2.0, 3, 0.000724064947895),
        (5, 1, 1, 1, 2, 2.0, 3, 0.0006569632791),
        (5, 1, 1, 2, 2, 2.0, 3, 0.000944074581746),
        (5, 1, 2, 1, 1, 10.0, 2, 0.0001739004672137),
        (5, 0, 1, 1, 1, 2.0, 3, 0.0010093514626),
        (5, 1, 2, 1, 1, 2.0, 3, 0.000707949219652),
        (5, 1, 2, 1, 1, 2.0, 4, 0.002702735512532),
        (5, 1, 2, 1, 1, 2.0, 5, 0.00292637099912),
        (5, 1, 1, 1, 3, 2.0, 2, 0.000254923743392),
        (5, 1, 1, 3, 1, 2.0, 2, 0.000499981827461),
    ],
)
def test_ordinal_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    embedding_dim,
    expected,
):
    """Test the ordinal transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = OrdinalTEEstimator(
        data_source,
        data_dest,
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        embedding_dim=embedding_dim,
    )
    res = est.result()
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected",
    [
        (5, 1, 0.0),
        (6, 1, 0.0),
        (5, 2, 0.001668221515),
        (6, 2, 0.002548207973),
        (7, 2, 0.001463248840),
        (5, 3, 0.000431291903),
        (6, 3, 0.000474948707),
        (5, 4, 0.000258042951),
        (6, 4, 0.000332053759),
        (5, 5, 5.2984961e-05),
        (6, 5, -3.0930624e-05),
    ],
)
def test_ordinal_cte(rng_int, embedding_dim, expected):
    """Test the conditional ordinal transfer entropy estimator."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = OrdinalCTEEstimator(
        data_source,
        data_dest,
        data_cond,
        embedding_dim=embedding_dim,
        base=2,
        stable=True,
    )
    res = est.result()
    if embedding_dim == 1:
        assert isinstance(res, float)
        assert res == 0.0
        return
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,step_size,src_hist_len,dest_hist_len,base,embedding_dim,expected",
    [
        (5, 1, 1, 1, 2.0, 2, 0.001668221515),
        (5, 2, 1, 1, 2.0, 2, 0.001499847155),
        (5, 3, 1, 1, 2.0, 2, 0.002246209913),
        (5, 1, 2, 1, 2.0, 3, 0.000854933772),
        (5, 1, 1, 2, 2.0, 3, 0.000463333827),
        (5, 1, 2, 2, 2.0, 3, 0.000741430819),
        (5, 2, 1, 1, 10.0, 2, 0.00045149898),
        (5, 1, 1, 1, 2.0, 3, 0.000431291903),
        (5, 2, 1, 1, 2.0, 3, 0.001090703247),
        (5, 2, 1, 1, 2.0, 4, 0.000759534290),
        (5, 2, 1, 1, 2.0, 5, 3.4400513e-05),
        (5, 1, 1, 3, 2.0, 2, 0.000199071124),
        (5, 1, 3, 1, 2.0, 2, 0.000749478055),
    ],
)
def test_ordinal_cte_slicing(
    rng_int,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    embedding_dim,
    expected,
):
    """Test the conditional ordinal transfer entropy estimator with slicing."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = OrdinalCTEEstimator(
        data_source,
        data_dest,
        data_cond,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        embedding_dim=embedding_dim,
        stable=True,
    )
    res = est.result()
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected_xy,expected_yx",
    [
        (1, 1, 0.0, 0.0),
        (1, 2, 0.00282503667, 0.000248359479),
        (1, 3, -0.00304940021, 0.000223400525),
        (1, 4, -0.00025159207, 0.00017453445),
        (1, 5, 0.00034294439, 0.000157359844),
        (2, 2, 0.00194385878, 9.45014940e-05),
        (2, 3, -0.00226919442, 0.000341760325),
        (3, 2, 0.00141154442, 0.000402597923),
        (3, 4, -0.00058057170, 0.000239034145),
    ],
)
def test_cte_ordinal_autoregressive(rng_int, embedding_dim, expected_xy, expected_yx):
    """Test the conditional ordinal transfer entropy estimator with
    autoregressive data."""
    data_source, data_dest, data_cond = discrete_random_variables_condition(rng_int)
    est_xy = OrdinalCTEEstimator(
        data_source,
        data_dest,
        data_cond,
        embedding_dim=embedding_dim,
        base=2,
        stable=True,
    )
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    est_yx = OrdinalCTEEstimator(
        data_dest,
        data_source,
        data_cond,
        embedding_dim=embedding_dim,
        base=2,
        stable=True,
    )
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)
