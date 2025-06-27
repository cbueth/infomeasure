"""Explicit tests for the Tsallis transfer entropy estimator."""

import pytest

from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)
from infomeasure.estimators.transfer_entropy import (
    TsallisTEEstimator,
    TsallisCTEEstimator,
)


@pytest.mark.parametrize(
    "k,q,expected",
    [
        ([2, 1.0, 0.1328256688984]),
        ([2, 1.1, 0.672315975388]),
        ([3, 2.0, 0.0168611881473]),
        ([1, 1.1, 0.604004850484]),
        ([2, 2.0, 0.015030160668]),
        ([4, 1.0, 0.06046248740893]),
    ],
)
def test_tsallis_te(k, q, expected):
    """Test the Tsallis transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(5, 0.5, 0.6, 0.4)
    est = TsallisTEEstimator(
        data_source,
        data_dest,
        k=k,
        q=q,
        noise_level=0,  # for reproducibility
        base=2,
    )
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,base,k,q,expected",
    [
        (5, 0, 1, 1, 1, 2.0, 2, 1.0, 0.1328256688984),
        (5, 0, 1, 1, 1, 2.0, 4, 1.0, 0.06046248740893),
        (5, 1, 1, 1, 1, 2.0, 4, 1.0, -0.00097675392276),
        (6, 0, 1, 1, 1, 2.0, 2, 1.0, 0.0978424366616),
        (5, 1, 2, 1, 1, 2.0, 2, 1.0, 0.02631960477116),
        (5, 1, 3, 1, 1, 2.0, 2, 1.0, -0.0707019638402),
        (5, 1, 1, 2, 1, 2.0, 2, 1.0, -0.03704319399725),
        (5, 1, 1, 1, 2, 2.0, 2, 1.0, -0.0350563983761),
        (5, 1, 1, 2, 2, 2.0, 2, 1.0, -0.0517794607390),
    ],
)
def test_tsallis_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    base,
    k,
    q,
    expected,
):
    """Test the Tsallis transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = TsallisTEEstimator(
        data_source,
        data_dest,
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        base=base,
        k=k,
        q=q,
        noise_level=0,  # for reproducibility
    )
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "k,q,expected",
    [
        ([2, 1.0, 0.1486385574]),
        ([2, 1.1, 0.4716667772]),
        ([3, 2.0, 0.0004727878]),
        ([1, 1.1, 0.4530811496]),
        ([2, 2.0, 0.0004229704]),
        ([4, 1.0, 0.1261881682]),
    ],
)
def test_tsallis_cte(k, q, expected):
    """Test the conditional Tsallis transfer entropy estimator."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        5, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = TsallisCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        k=k,
        q=q,
        noise_level=0,  # for reproducibility
        base=2,
    )
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,step_size,src_hist_len,dest_hist_len,k,q,expected",
    [
        (5, 1, 1, 1, 2, 1.0, 0.103028397),
        (5, 1, 1, 1, 4, 1.0, 0.087466973),
        (6, 1, 1, 1, 2, 1.0, 0.091019176),
        (5, 2, 1, 1, 2, 1.0, 0.064470937),
        (5, 3, 1, 1, 2, 1.0, -0.06503512),
        (5, 1, 2, 1, 2, 1.0, 0.051826538),
        (5, 1, 1, 2, 2, 1.0, 0.089287281),
        (5, 1, 2, 2, 2, 1.0, -0.00521769),
    ],
)
def test_tsallis_cte_slicing(
    rng_int,
    step_size,
    src_hist_len,
    dest_hist_len,
    k,
    q,
    expected,
):
    """Test the conditionalTsallis transfer entropy estimator with slicing."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = TsallisCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        k=k,
        q=q,
        noise_level=0,  # for reproducibility
    )
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "rng_int,method,p_te,p_cte",
    [
        (1, "permutation_test", 0.0, 0.0),
        (1, "bootstrap", 0.0, 0.0),
        (2, "permutation_test", 0.02, 0.0),
        (2, "bootstrap", 0.0, 0.0),
        (3, "permutation_test", 0.0, 0.0),
        (3, "bootstrap", 0.0, 0.0),
        (4, "permutation_test", 0.0, 0.0),
        (4, "bootstrap", 0.0, 0.0),
    ],
)
def test_tsallis_te_statistical_test(rng_int, method, p_te, p_cte):
    """Test the Tsallis TE for p-values. Fix rng."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est_te_xy = TsallisTEEstimator(
        data_source, data_dest, k=4, q=1.0, noise_level=0, base=2, seed=8
    )
    est_cte_xy = TsallisCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        k=4,
        q=1.0,
        noise_level=0,
        base=2,
        seed=8,
    )
    test = est_te_xy.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_te)
    test = est_cte_xy.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_cte)


@pytest.mark.parametrize(
    "rng_int,method,eff_te,eff_cte",
    [
        (1, "permutation_test", 0.15379927286749506, 0.0),
        (1, "bootstrap", 0.161452619631973, 0.0),
        (2, "permutation_test", 0.06279044408929124, 0.0),
        (2, "bootstrap", 0.09545814052331991, 0.0),
        (3, "permutation_test", 0.095582966279391, 0.0),
        (4, "permutation_test", 0.1138150515118923, 0.0),
    ],
)
def test_tsallis_te_effective_val(rng_int, method, eff_te, eff_cte):
    """Test the Tsallis transfer entropy for effective values. Fix rng."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est_te_xy = TsallisTEEstimator(
        data_source, data_dest, k=4, q=1.0, noise_level=0, base=2, seed=8
    )
    est_cte_xy = TsallisCTEEstimator(
        data_source,
        data_dest,
        cond=data_source,
        k=4,
        q=1.0,
        noise_level=0,
        base=2,
        seed=8,
    )
    assert est_te_xy.effective_val(method=method) == pytest.approx(eff_te)
    assert est_cte_xy.effective_val(method=method) == pytest.approx(eff_cte)
