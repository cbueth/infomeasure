"""Explicit symbolic / permutation mutual information estimator."""

import pytest
from numpy import isnan

from infomeasure.measures.mutual_information import SymbolicMIEstimator


@pytest.mark.parametrize("data_len", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("order", [1, 2, 5])
@pytest.mark.parametrize("per_symbol", [True, False])
@pytest.mark.parametrize("step_size", [1, 2, 3])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_symbolic_entropy(data_len, order, per_symbol, step_size, offset, default_rng):
    """Test the discrete entropy estimator."""
    data_x = default_rng.integers(0, 10, data_len)
    data_y = default_rng.integers(0, 10, data_len)
    if data_len - abs(offset) < (order - 1) * step_size + 1:
        with pytest.raises(ValueError):
            est = SymbolicMIEstimator(
                data_x,
                data_y,
                order,
                per_symbol=per_symbol,
                step_size=step_size,
                offset=offset,
            )
            est.results()
        return
    if order == 1:
        est = SymbolicMIEstimator(
            data_x,
            data_y,
            order,
            per_symbol=per_symbol,
            step_size=step_size,
            offset=offset,
        )
        assert est.global_val() == 0
        for i in est.local_val():
            assert i == 0
        assert isnan(est.std_val())
        return
    est = SymbolicMIEstimator(
        data_x,
        data_y,
        order,
        per_symbol=per_symbol,
        step_size=step_size,
        offset=offset,
    )
    max_val = est._log_base(data_len)
    assert 0 <= est.global_val() <= max_val
    for i in est.local_val():
        assert isinstance(i, float)
    assert 0 <= est.std_val() <= max_val


@pytest.mark.parametrize("order", [-1, 1.0, "a", 1.5, 2.0])
def test_symbolic_entropy_invalid_order(order, default_rng):
    """Test the discrete entropy estimator with invalid order."""
    data = list(range(10))
    with pytest.raises(ValueError):
        SymbolicMIEstimator(data, data, order)
