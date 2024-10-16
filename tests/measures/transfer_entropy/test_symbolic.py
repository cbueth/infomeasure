"""Explicit symbolic / permutation transfer entropy estimator."""

import pytest
from numpy import isnan

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
