"""Explicit ordinal / permutation transfer entropy estimator."""

import pytest
from numpy import isnan, e, log, asarray

from infomeasure import te
from infomeasure.estimators.transfer_entropy import (
    OrdinalCTEEstimator,
    OrdinalTEEstimator,
)
from tests.conftest import (
    discrete_random_variables_condition,
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
    discrete_random_variables_shifted,
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
        for i in est.local_vals():
            assert i == 0
        if len(est.local_vals()) > 0:
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
        (5, 2, 0.0257169029),
        (6, 2, 0.0179241626),
        (7, 2, 0.0347067536),
        (5, 3, 0.1090099579),
        (6, 3, 0.0841433146),
        (5, 4, 0.9282273267),
        (6, 4, 0.9816978506),
        (5, 5, 1.6187348457),
        (6, 5, 1.6550450501),
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
    est.local_vals()


@pytest.mark.parametrize(
    "rng_int,embedding_dim,prop_time,base,expected_xy,expected_xc",
    [
        (1, 2, 0, 2, 0.87707321, 0.00058214135),
        (1, 2, 1, 2, 0.87649858, 0.00066425255),
        (1, 3, 1, 2, 1.4169192, 0.06293442),
        (1, 4, 1, 2, 1.8091009, 0.78871896),
        (1, 5, 1, 2, 1.7659465, 1.6169571),
        (1, 6, 1, 2, 0.98537632, 1.0487262),
        (1, 7, 1, 2, 0.30596859, 0.39661209),
        (1, 8, 1, 2, 0.081406137, 0.10395102),
        (1, 2, 2, 2, 0.87624404, 0.0022847915),
        (1, 2, 3, 2, 0.87598611, 0.00037429202),
        (2, 2, 1, 2, 0.9080654, 2.06959e-05),
        (2, 2, 1, e, 0.62942297, 1.4345305e-05),
        (2, 2, 1, 10, 0.27335492, 6.2300867e-06),
        (3, 2, 1, 2, 0.91241141, 0.00061093181),
        (4, 2, 1, 2, 0.88478851, 0.00024895695),
        (5, 2, 1, 2, 0.88559227, 0.0060727024),
        (6, 2, 1, 2, 0.87798884, 0.0014716052),
    ],
)
def test_te_discrete_shifted(
    rng_int, embedding_dim, prop_time, base, expected_xy, expected_xc
):
    """Test the discrete transfer entropy estimator with shifted data."""
    source, dest, control = discrete_random_variables_shifted(
        rng_int, prop_time, low=0, high=5
    )
    est_xy = OrdinalTEEstimator(
        source,
        dest,
        base=base,
        prop_time=prop_time - 1,
        embedding_dim=embedding_dim,
        stable=True,
    )
    res_xy = est_xy.result()
    assert isinstance(res_xy, float)
    assert res_xy == pytest.approx(expected_xy)
    est_xc = OrdinalTEEstimator(
        source,
        control,
        base=base,
        prop_time=prop_time - 1,
        embedding_dim=embedding_dim,
        stable=True,
    )
    res_xc = est_xc.result()
    assert isinstance(res_xc, float)
    assert res_xc == pytest.approx(expected_xc)
    est_xy.local_vals()
    est_xc.local_vals()


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,base,embedding_dim,expected",
    [
        (5, 0, 1, 1, 1, 2.0, 2, 0.02571690291),
        (5, 1, 2, 1, 1, 2.0, 2, 0.0001610875315),
        (5, 1, 3, 1, 1, 2.0, 2, 0.001006379651),
        (5, 1, 1, 2, 1, 2.0, 3, 0.2020141204),
        (5, 1, 1, 1, 2, 2.0, 3, 0.1826357915),
        (5, 1, 1, 2, 2, 2.0, 3, 0.4956391554),
        (5, 1, 2, 1, 1, 10.0, 2, 4.84921789e-05),
        (5, 0, 1, 1, 1, 2.0, 3, 0.1090099579),
        (5, 1, 2, 1, 1, 2.0, 3, 0.08746380612),
        (5, 1, 2, 1, 1, 2.0, 4, 1.254118058),
        (5, 1, 2, 1, 1, 2.0, 5, 1.407790949),
        (5, 1, 1, 1, 3, 2.0, 2, 0.00815755978),
        (5, 1, 1, 3, 1, 2.0, 2, 0.0159994184),
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
        stable=True,
    )
    res = est.result()
    assert res == pytest.approx(expected)
    est.local_vals()


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected",
    [
        (5, 1, 0.0),
        (6, 1, 0.0),
        (5, 2, 0.0266969890),
        (6, 2, 0.0407713275),
        (7, 2, 0.0244538839),
        (5, 3, 0.354216979),
        (6, 3, 0.344597243),
        (5, 4, 0.809789189),
        (6, 4, 0.811645250),
        (5, 5, 0.0778992713),
        (6, 5, 0.0866981532),
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
        cond=data_cond,
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
    est.local_vals()


@pytest.mark.parametrize(
    "rng_int,step_size,src_hist_len,dest_hist_len,base,embedding_dim,expected",
    [
        (5, 1, 1, 1, 2.0, 2, 0.026696989),
        (5, 2, 1, 1, 2.0, 2, 0.010281364),
        (5, 3, 1, 1, 2.0, 2, 0.018216100),
        (5, 1, 2, 1, 2.0, 3, 0.76446151),
        (5, 1, 1, 2, 2.0, 3, 0.60662724),
        (5, 1, 2, 2, 2.0, 3, 0.95203964),
        (5, 2, 1, 1, 10.0, 2, 0.0030949991),
        (5, 1, 1, 1, 2.0, 3, 0.3542169),
        (5, 2, 1, 1, 2.0, 3, 0.5646381),
        (5, 2, 1, 1, 2.0, 4, 0.5367281),
        (5, 2, 1, 1, 2.0, 5, 4 / 165),
        (5, 1, 1, 3, 2.0, 2, 0.03516579),
        (5, 1, 3, 1, 2.0, 2, 0.04797205),
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
        cond=data_cond,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        embedding_dim=embedding_dim,
        stable=True,
    )
    res = est.result()
    assert res == pytest.approx(expected)
    est.local_vals()


@pytest.mark.parametrize(
    "rng_int,embedding_dim,expected_xy,expected_yx",
    [
        (1, 1, 0.0, 0.0),
        (1, 2, 0.04530477596, 0.00444690475),
        (1, 3, 0.2198833229, 0.187374397),
        (1, 4, 0.8567380900, 0.564197118),
        (1, 5, 0.6287293606, 0.162603467),
        (2, 2, 0.03125016321, 0.00198517698),
        (2, 3, 0.2451663233, 0.204250588),
        (3, 2, 0.02267760186, 0.00646406629),
        (3, 4, 0.8442503912, 0.583530971),
    ],
)
def test_cte_ordinal_autoregressive(rng_int, embedding_dim, expected_xy, expected_yx):
    """Test the conditional ordinal transfer entropy estimator with
    autoregressive data."""
    data_source, data_dest, data_cond = discrete_random_variables_condition(rng_int)
    est_xy = OrdinalCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
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
        cond=data_cond,
        embedding_dim=embedding_dim,
        base=2,
        stable=True,
    )
    res_yx = est_yx.result()
    assert isinstance(res_yx, float)
    assert res_yx == pytest.approx(expected_yx)


@pytest.mark.parametrize(
    "data",
    [
        ([1, 2], [[1, 2], [3, 4]]),
        ([(1, 2), (3, 4)], [1, 2]),
        ([1, 2, 3], [[1, 1], [2, 2], [3, 3]]),
        ([1, 2, 3], [[1], [2], [3]]),
        ([1, 2], [[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_ordinal_TE_invalid_data_type(data):
    """Test the ordinal TE estimator with invalid data type."""
    with pytest.raises(
        TypeError,
        match="The data must be tuples of 1D arrays. "
        "Ordinal patterns can only be computed from 1D arrays.",
    ):
        # est = OrdinalTEEstimator(*data, embedding_dim=2)
        # est.result()
        te(*data, approach="ordinal", embedding_dim=2)


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
    ],
)
def test_ordinal_CTE_invalid_data_type(data, cond):
    """Test the ordinal CTE estimator with invalid data type."""
    with pytest.raises(
        TypeError,
        match=(
            "The data must be tuples of 1D arrays. "
            "Ordinal patterns can only be computed from 1D arrays."
        )
        if asarray(cond).ndim == 1
        else "The conditional variable must be an 1d array,",
    ):
        est = OrdinalCTEEstimator(*data, cond=cond, embedding_dim=2)
        est.result()


@pytest.mark.parametrize(
    "data",
    [
        ([1, 2, 3, 4], [None] * 4),  # Incomparable
        ([1, 2, 3, 4], [1, 2, None, None]),  # Inhomogeneous
    ],
)
@pytest.mark.parametrize("embedding_dim", [2, 3, 4])
def test_ordinal_te_type_incomparable(data, embedding_dim):
    """Test the ordinal TE estimator with incomparable data type."""
    with pytest.raises(
        TypeError,
        match="'<' not supported between instances of",
    ):
        est = OrdinalTEEstimator(*data, embedding_dim=embedding_dim)
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
def test_ordinal_cte_type_incomparable(data, cond, embedding_dim):
    """Test the ordinal CTE estimator with incomparable data type."""
    with pytest.raises(
        TypeError,
        match="'<' not supported between instances of",
    ):
        est = OrdinalCTEEstimator(*data, cond=cond, embedding_dim=embedding_dim)
        est.result()
