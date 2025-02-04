"""Explicit symbolic / permutation transfer entropy estimator."""

import pytest
from numpy import isnan, ndarray, std

from tests.conftest import generate_autoregressive_series
from infomeasure.measures.transfer_entropy import SymbolicTEEstimator


@pytest.mark.parametrize("data_len", [1, 2, 3, 10, 100])
@pytest.mark.parametrize("order", [2, 3, 5])
@pytest.mark.parametrize("step_size", [1, 2, 3])
@pytest.mark.parametrize("prop_time", [0, 1, 4])
def test_symbolic_entropy(data_len, order, step_size, prop_time, default_rng):
    """Test the discrete entropy estimator."""
    source = default_rng.integers(0, 10, data_len)
    dest = default_rng.integers(0, 10, data_len)
    if data_len - abs(prop_time * step_size) <= (order - 1) * step_size + 1:
        with pytest.raises(ValueError):
            est = SymbolicTEEstimator(
                source,
                dest,
                order,
                step_size=step_size,
                prop_time=prop_time,
            )
            est.results()
        return
    if order == 1:
        est = SymbolicTEEstimator(
            source,
            dest,
            order,
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
    est = SymbolicTEEstimator(
        source,
        dest,
        order,
        step_size=step_size,
        prop_time=prop_time,
    )
    max_val = est._log_base(data_len)
    assert 0 <= est.global_val() <= max_val
    for i in est.local_val():
        assert isinstance(i, float)
    if len(est.local_val()) > 0:
        assert 0 <= est.std_val() <= max_val
    else:
        assert isnan(est.std_val())


@pytest.mark.parametrize("order", [-1, 1.0, "a", 1.5, 2.0])
def test_symbolic_entropy_invalid_order(order, default_rng):
    """Test the discrete entropy estimator with invalid order."""
    data = list(range(10))
    with pytest.raises(ValueError):
        SymbolicTEEstimator(data, data, order)


@pytest.mark.parametrize(
    "rng_int,order,expected",
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
def test_symbolic_te(rng_int, order, expected):
    """Test the symbolic transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = SymbolicTEEstimator(data_source, data_dest, order, base=2)
    res = est.results()
    if order == 1:
        assert isinstance(res, float)
        assert res == 0.0
        return
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert isinstance(res[1], ndarray)
    assert isinstance(res[2], float)
    assert res[0] == pytest.approx(expected)
    assert res[2] == pytest.approx(std(res[1]))


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,base,order,expected",
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
def test_symbolic_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    order,
    expected,
):
    """Test the symbolic transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = SymbolicTEEstimator(
        data_source,
        data_dest,
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        order=order,
    )
    res = est.results()
    assert isinstance(res, tuple)
    assert len(res) == 3
    assert isinstance(res[0], float)
    assert isinstance(res[1], ndarray)
    assert isinstance(res[2], float)
    assert res[0] == pytest.approx(expected)
    if order == 1:
        assert isnan(res[2])
    else:
        assert res[2] == pytest.approx(std(res[1]))
