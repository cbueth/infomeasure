"""Explicit symbolic / permutation mutual information estimator tests."""

from datetime import datetime

import pytest
from numpy import ndarray

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.estimators.mutual_information import (
    SymbolicMIEstimator,
    SymbolicCMIEstimator,
)


@pytest.mark.parametrize("data_len", [10, 100, 1000])
@pytest.mark.parametrize("order", [1, 2, 5])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_symbolic_mi(data_len, order, offset, default_rng):
    """Test the discrete mutual information estimator."""
    data_x = default_rng.integers(0, 10, data_len)
    data_y = default_rng.integers(0, 10, data_len)
    if data_len - abs(offset) < (order - 1) + 1:
        with pytest.raises(ValueError):
            est = SymbolicMIEstimator(
                data_x,
                data_y,
                order=order,
                offset=offset,
            )
            est.global_val()
        return
    est = SymbolicMIEstimator(
        data_x,
        data_y,
        order=order,
        offset=offset,
    )

    if order == 1:
        assert est.global_val() == 0.0  # no local values returned
    else:
        max_val = est._log_base(data_len)
        assert 0 <= est.global_val() <= max_val
    assert isinstance(est.local_val(), ndarray)


@pytest.mark.parametrize("order", [-1, 1.0, "a", 1.5, 2.0])
def test_symbolic_mi_invalid_order(order, default_rng):
    """Test the discrete mutual information estimator with invalid order."""
    data = list(range(10))
    with pytest.raises(ValueError):
        SymbolicMIEstimator(data, data, order=order)


@pytest.mark.parametrize(
    "data_x,data_y,order,expected",
    [
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], 1, 0.0),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], 2, 0.8112781244591328),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], 3, 1.584962500721156),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], 4, 1.0),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], 5, 0.0),
        (
            [0.74, 0.64, 0.03, 0.67, 0.84, 0.3, 0.84, 0.62, 0.79, 0.28],
            [0.25, 0.65, 0.05, 0.73, 0.57, 0.31, 0.71, 0.7, 0.59, 0.26],
            2,
            0.07278022578373262,
        ),
        (
            [0.74, 0.64, 0.03, 0.67, 0.84, 0.3, 0.84, 0.62, 0.79, 0.28],
            [0.25, 0.65, 0.05, 0.73, 0.57, 0.31, 0.71, 0.7, 0.59, 0.26],
            3,
            1.9056390622295665,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            2,
            0.07278022578373262,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            3,
            1.2169171866886992,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            4,
            2.8073549220576046,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            5,
            2.584962500721156,
        ),
    ],
)
def test_symbolic_mi_values(data_x, data_y, order, expected):
    """Test the symbolic mutual information estimator."""
    est = SymbolicMIEstimator(data_x, data_y, order=order, base=2, stable=True)
    assert est.global_val() == pytest.approx(expected)
    assert isinstance(est.local_val(), ndarray)


@pytest.mark.parametrize(
    "rng_int,order,expected",
    [
        (5, 1, 0.0),
        (5, 2, 0.004881048),
        (5, 3, 0.047311377),
        (5, 4, 0.535645231),
        (5, 5, 3.42514917),
        (6, 2, 0.001331982),
        (6, 3, 0.037669081),
        (6, 4, 0.470012588),
        (7, 3, 0.080807403),
        (7, 4, 0.604484529),
    ],
)
def test_symbolic_mi_values_autoregressive(rng_int, order, expected):
    """Test the symbolic mutual information estimator with autoregressive data."""
    data_x, data_y = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = SymbolicMIEstimator(data_x, data_y, order=order, base=2, stable=True)
    res = est.result()
    assert isinstance(est.result(), float)
    assert res == pytest.approx(expected)
    assert isinstance(est.local_val(), ndarray)


@pytest.mark.parametrize(
    "rng_int,order,expected",
    [
        (5, 1, 0.0),
        (5, 2, 0.003364152),
        (5, 3, 0.160698565),
        (5, 4, 2.717637411),
        (5, 5, 2.93783424),
        (6, 2, 0.006491342),
        (6, 3, 0.145752994),
        (6, 4, 2.675821510),
        (7, 3, 0.153249279),
        (7, 4, 2.702059925),
    ],
)
def test_symbolic_mi_values_autoregressive_condition(rng_int, order, expected):
    """Test the symbolic mutual information estimator with autoregressive data."""
    data_x, data_y, cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = SymbolicCMIEstimator(
        data_x, data_y, cond=cond, order=order, base=2, stable=True
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_val(), ndarray)


@pytest.mark.parametrize(
    "data_x,data_y,cond,order,expected",
    [
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], [0, 2, 0, 1, 0], 1, 0.0),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], [0, 2, 0, 1, 0], 2, 1 / 2),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], [0, 2, 0, 1, 0], 3, 2 / 3),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], [0, 1, 0, 1, 2], 4, 0.0),
        ([0, 1, 0, 1, 2], [0, 1, 0, 1, 2], [0, 2, 0, 1, 0], 5, 0.0),
        (
            [0.74, 0.64, 0.03, 0.67, 0.84, 0.3, 0.84, 0.62, 0.79, 0.28],
            [0.25, 0.65, 0.05, 0.73, 0.57, 0.31, 0.71, 0.7, 0.59, 0.26],
            [0.26, 0.65, 0.05, 0.73, 0.31, 0.71, 0.7, 0.59, 0.57, 0.25],
            2,
            0.211126058876,
        ),
        (
            [0.74, 0.64, 0.03, 0.67, 0.84, 0.3, 0.84, 0.62, 0.79, 0.28],
            [0.25, 0.65, 0.05, 0.73, 0.57, 0.31, 0.71, 0.7, 0.59, 0.26],
            [0.26, 0.65, 0.05, 0.73, 0.31, 0.71, 0.7, 0.59, 0.57, 0.25],
            3,
            0.59436093777,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            [0.26, 0.65, 0.05, 0.73, 0.31, 0.71, 0.7, 0.59, 0.57, 0.25],
            2,
            0.211126058876,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            [0.26, 0.65, 0.05, 0.73, 0.31, 0.71, 0.7, 0.59, 0.57, 0.25],
            3,
            0.59436093777,
        ),
        (
            [0.78, 0.92, 0.96, 0.16, 0.13, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            [0.26, 0.65, 0.05, 0.73, 0.31, 0.71, 0.7, 0.59, 0.57, 0.25],
            4,
            0.285714285714,
        ),
        (
            [0.78, 0.92, 0.13, 0.96, 0.16, 0.03, 0.95, 0.44, 0.27, 0.33],
            [0.39, 0.72, 0.54, 0.97, 0.61, 0.06, 0.85, 0.8, 0.64, 0.41],
            [0.26, 0.65, 0.05, 0.73, 0.31, 0.71, 0.7, 0.59, 0.57, 0.25],
            5,
            0.0,
        ),
    ],
)
def test_symbolic_cmi_values(data_x, data_y, cond, order, expected):
    """Test the symbolic conditional mutual information estimator."""
    est = SymbolicCMIEstimator(
        data_x, data_y, order=order, cond=cond, base=2, stable=True
    )
    assert est.global_val() == pytest.approx(expected)
    est.local_val()  # Checks internally for `global = mean(local)`
