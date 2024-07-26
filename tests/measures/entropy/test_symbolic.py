"""Explicit symbolic / permutation entropy estimator tests."""

import pytest

from infomeasure.measures.entropy import SymbolicEntropyEstimator


@pytest.mark.parametrize("data_len", [1, 2, 10, 100, 1000])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("per_symbol", [True, False])
def test_symbolic_entropy(data_len, order, per_symbol, default_rng):
    """Test the discrete entropy estimator."""
    data = default_rng.integers(0, 10, data_len)
    if order == 1:
        est = SymbolicEntropyEstimator(data, order=order, per_symbol=per_symbol)
        assert est.results() == 0
        return
    if order > data_len:
        with pytest.raises(ValueError):
            est = SymbolicEntropyEstimator(data, order=order, per_symbol=per_symbol)
            est.results()
        return
    est = SymbolicEntropyEstimator(data, order=order, per_symbol=per_symbol)
    assert 0 <= est.results() <= est._log_base(data_len)


@pytest.mark.parametrize(
    "data,order,base,expected",
    [
        ([0, 1, 0, 1, 0], 2, 2, 1.0),  # 2x(01), 2x(10): log_2(2) = 1
        ([0, 2, 4, 3, 1], 3, 3, 1.0),  # 1x(012), 1x(021), 1x(210): log_3(3) = 1
        ([0, 2, 4, 3, 1], 3, 2, 1.584962500721156),
        # 1x(012), 1x(021), 1x(210): log_2(3) = 1.584962500721156
        ([0, 1, 0, 1, 2, 0], 2, 2, 0.9709505944546686),
        # 3x(01), 2x(10): 3/5*log_2(5/3) + 2/5*log_2(5/2) = 0.9709505944546686
        (list(range(10)), 3, 2, 0.0),  # 8x(012): log_2(1) = 0
        (list(range(10)), 3, "e", 0.0),  # 8x(012): log_e(1) = 0
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 2, 2, 0.9798687566511528),
        # 7x(01), 5x(10): 7/12*log_2(12/7) + 5/12*log_2(12/5) = 0.9798687566511528
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 2, "e", 0.6791932659915256),
        # 7x(01), 5x(10): 7/12*log_e(12/7) + 5/12*log_e(12/5) = 0.6791932659915256
        ([4, 7, 9, 10, 6, 11, 3], 3, 2, 1.5219280948873621),
        # 2x(012), 2x(201), (102): 2*2/5*log_2(5/2) + 1*1/5*log_2(5/1) = 1.5219280948873621
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 3, 2, 2.481714572986073),
        # 3x(021), 2x(120), 2x(012), (210), 2x(102), (201): 1*3/11*log_2(11/3) + 3*2/11*log_2(11/2) + 2*1/11*log_2(11/1) = 2.481714572986073
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 5, 2, 3.169925001442312),
        # ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 12, 2, 1.0),  # TODO: not compatible yet, as code fails trying to generate all 12! combinations
        (["a", "b", "a", "b", "a"], 2, 2, 1.0),  # 2x(10), 2x(0,1): log_2(2) = 1
        ([0.0, 1.0, 0.0, 1.0, 0.0], 2, 2, 1.0),  # 2x(0,1), 2x(10): log_2(2) = 1
    ],
)
def test_symbolic_entropy_explicit(data, order, base, expected):
    """Test the discrete entropy estimator with explicit values."""
    assert SymbolicEntropyEstimator(
        data, order=order, base=base
    ).results() == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("order", [-1, 1.0, "a", 1.5, 2.0])
def test_symbolic_entropy_invalid_order(order, default_rng):
    """Test the discrete entropy estimator with invalid order."""
    data = list(range(10))
    with pytest.raises(ValueError):
        SymbolicEntropyEstimator(data, order=order)
