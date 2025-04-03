"""Explicit ordinal / permutation mutual information estimator tests."""

from datetime import datetime

import pytest
from numpy import ndarray, asarray

from infomeasure.estimators.entropy import OrdinalEntropyEstimator
from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.estimators.mutual_information import (
    OrdinalMIEstimator,
    OrdinalCMIEstimator,
)


@pytest.mark.parametrize("data_len", [10, 100, 1000])
@pytest.mark.parametrize("embedding_dim", [1, 2, 5])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_ordinal_mi(data_len, embedding_dim, offset, default_rng):
    """Test the discrete mutual information estimator."""
    data_x = default_rng.integers(0, 10, data_len)
    data_y = default_rng.integers(0, 10, data_len)
    if data_len - abs(offset) < (embedding_dim - 1) + 1:
        with pytest.raises(ValueError):
            est = OrdinalMIEstimator(
                data_x,
                data_y,
                embedding_dim=embedding_dim,
                offset=offset,
            )
            est.global_val()
        return
    est = OrdinalMIEstimator(
        data_x,
        data_y,
        embedding_dim=embedding_dim,
        offset=offset,
    )

    if embedding_dim == 1:
        assert est.global_val() == 0.0  # no local values returned
    else:
        max_val = est._log_base(data_len)
        assert 0 <= est.global_val() <= max_val
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize("embedding_dim", [-1, 1.0, "a", 1.5, 2.0])
def test_ordinal_mi_invalid_embedding_dim(embedding_dim, default_rng):
    """Test the discrete mutual information estimator with invalid embedding_dim."""
    data = list(range(10))
    with pytest.raises(ValueError):
        OrdinalMIEstimator(data, data, embedding_dim=embedding_dim)


@pytest.mark.parametrize(
    "data_x,data_y,embedding_dim,expected",
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
def test_ordinal_mi_values(data_x, data_y, embedding_dim, expected):
    """Test the ordinal mutual information estimator."""
    est = OrdinalMIEstimator(
        data_x, data_y, embedding_dim=embedding_dim, base=2, stable=True
    )
    assert est.global_val() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected",
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
def test_ordinal_mi_values_autoregressive(rng_int, embedding_dim, expected):
    """Test the ordinal mutual information estimator with autoregressive data."""
    data_x, data_y = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = OrdinalMIEstimator(
        data_x, data_y, embedding_dim=embedding_dim, base=2, stable=True
    )
    res = est.result()
    assert isinstance(est.result(), float)
    assert res == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected",
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
def test_ordinal_mi_values_autoregressive_condition(rng_int, embedding_dim, expected):
    """Test the ordinal mutual information estimator with autoregressive data."""
    data_x, data_y, cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = OrdinalCMIEstimator(
        data_x, data_y, cond=cond, embedding_dim=embedding_dim, base=2, stable=True
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "data_x,data_y,cond,embedding_dim,expected",
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
def test_ordinal_cmi_values(data_x, data_y, cond, embedding_dim, expected):
    """Test the ordinal conditional mutual information estimator."""
    est = OrdinalCMIEstimator(
        data_x, data_y, embedding_dim=embedding_dim, cond=cond, base=2, stable=True
    )
    assert est.global_val() == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize("rng_int", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("emb_dim", [1, 2, 3, 4, 8])
def test_entropy_equality(rng_int, default_rng, emb_dim):
    """Test the equality of MI(x, x) = H(x)."""
    x = default_rng.normal(scale=1, size=1000)
    mutual_info = OrdinalMIEstimator(x, x, embedding_dim=emb_dim, base="e")
    entropy = OrdinalEntropyEstimator(x, embedding_dim=emb_dim, base="e")
    assert entropy.result() == pytest.approx(mutual_info.result())


@pytest.mark.parametrize(
    "data",
    [
        ([1, 2], [[1, 2], [3, 4]]),
        ([(1, 2), (3, 4)], [1, 2]),
        ([1, 2, 3], [[1, 1], [2, 2], [3, 3]]),
        ([1, 2, 3], [[1], [2], [3]]),
        ([1, 2], [[1, 2, 3], [4, 5, 6]]),
        ([1, 2], [1, 2], [1, 2], [[1, 1], [2, 2]]),
    ],
)
def test_ordinal_MI_invalid_data_type(data):
    """Test the ordinal MI estimator with invalid data type."""
    with pytest.raises(
        TypeError,
        match="The data must be tuples of 1D arrays. "
        "Ordinal patterns can only be computed from 1D arrays.",
    ):
        est = OrdinalMIEstimator(*data, embedding_dim=2)
        est.result()


@pytest.mark.parametrize(
    "data, cond",
    [
        [([1, 2], [[1, 2], [3, 4]]), [1, 2]],
        [([1, 2], [1, 2]), [[1], [2]]],
        [([1, 2], [1, 2]), [[1, 2], [3, 4]]],
        [([1, 2], [1, 2]), [(1, 2), (3, 4)]],
        [([1, 2, 3], [[1, 1], [2, 2], [3, 3]]), [1, 2, 3]],
        [([1, 2, 3], [1, 2, 3]), [[1, 1], [2, 2], [3, 3]]],
        [([1, 2, 3], [[1], [2], [3]]), [1, 2, 3]],
        [([1, 2, 3], [1, 2, 3]), [[1], [2], [3]]],
        [([1, 2], [[1, 1, 1], [2, 2, 2]]), [1, 2]],
        [([1, 2], [1, 2]), [[1, 1, 1], [2, 2, 2]]],
        [([1, 2], [1, 2], [1, 2], [[1, 1], [2, 2]]), [1, 2]],
        [([1, 2], [1, 2], [1, 2], [1, 2]), [[1, 1], [2, 2]]],
    ],
)
def test_ordinal_CMI_invalid_data_type(data, cond):
    """Test the ordinal CMI estimator with invalid data type."""
    with pytest.raises(
        TypeError,
        match=(
            "The data must be tuples of 1D arrays. "
            "Ordinal patterns can only be computed from 1D arrays."
        )
        if asarray(cond).ndim == 1
        else "The conditional variable must be an 1d array,",
    ):
        est = OrdinalCMIEstimator(*data, cond=cond, embedding_dim=2)
        est.result()


@pytest.mark.parametrize(
    "data",
    [
        ([1, 2, 3, 4], [None] * 4),  # Incomparable
        ([1, 2, 3, 4], [1, 2, None, None]),  # Inhomogeneous
    ],
)
@pytest.mark.parametrize("embedding_dim", [2, 3, 4])
def test_ordinal_mi_type_incomparable(data, embedding_dim):
    """Test the ordinal MI estimator with incomparable data type."""
    with pytest.raises(
        TypeError,
        match="'<' not supported between instances of",
    ):
        est = OrdinalMIEstimator(*data, embedding_dim=embedding_dim)
        est.result()


@pytest.mark.parametrize(
    "data,cond",
    [
        [([1, 2, 3, 4], [None] * 4), [1, 2, 3, 4]],  # Incomparable
        [([1, 2, 3, 4], [1, 2, 3, 4]), [None] * 4],  # Incomparable
        [([1, 2, 3, 4], [1, 2, None, None]), [1, 2, 3, 4]],  # Inhomogeneous
        [([1, 2, 3, 4], [1, 2, 3, 4]), [None] * 4],  # Inhomogeneous
    ],
)
@pytest.mark.parametrize("embedding_dim", [2, 3, 4])
def test_ordinal_cmi_type_incomparable(data, cond, embedding_dim):
    """Test the ordinal CMI estimator with incomparable data type."""
    with pytest.raises(
        TypeError,
        match="'<' not supported between instances of",
    ):
        est = OrdinalCMIEstimator(*data, cond=cond, embedding_dim=embedding_dim)
        est.result()
