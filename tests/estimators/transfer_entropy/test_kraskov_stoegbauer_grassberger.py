"""
Explicit tests for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator.
"""

import numpy as np
import pytest
from numpy import ndarray, inf

from infomeasure import transfer_entropy
from infomeasure.estimators.transfer_entropy import KSGTEEstimator, KSGCTEEstimator
from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
)


@pytest.mark.parametrize(
    "rng_int,k,ksg_id,minkowski_p,base,expected",
    [
        (5, 4, 1, 2, "e", -0.1464972645512768),
        (5, 4, 1, 3, "e", -0.07711205666343),
        (5, 4, 2, 3, "e", -0.32711205666),
        (5, 4, 1, inf, "e", 0.09269348225),
        (5, 16, 1, 2, "e", -0.09411416419535731),
        (5, 16, 1, 3, "e", -0.018089069918),
        (5, 16, 1, inf, "e", 0.1013825511),
        (5, 16, 2, 2, "e", -0.1566141641),
        (5, 16, 2, 3, "e", -0.080589069918),
        (5, 16, 2, inf, "e", 0.0126694050),
        (6, 4, 1, 2, "e", -0.14070180426292878),
        (6, 4, 1, 3, "e", -0.076275290026),
        (6, 4, 1, inf, "e", 0.0765195523),
        (7, 4, 1, 2, "e", -0.1333354231111923),
        (7, 4, 1, 3, "e", -0.07016025200),
        (7, 4, 1, inf, "e", 0.09183638914294),
        (7, 4, 1, 2, 2.0, -0.19236235),
        (7, 4, 1, 3, 10, -0.03047021),
        (7, 4, 1, inf, 5, 0.057061156),
    ],
)
def test_ksg_te(rng_int, k, ksg_id, minkowski_p, base, expected):
    """Test the KSG transfer entropy estimator."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KSGTEEstimator(
        data_source,
        data_dest,
        k=k,
        ksg_id=ksg_id,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,prop_time,step_size,src_hist_len,dest_hist_len,expected",
    [
        (5, 1, 1, 1, 1, -0.005709180172),
        (6, 1, 1, 1, 1, 0.008991300793),
        (7, 1, 1, 1, 1, 0.008242300136),
        (5, 0, 1, 1, 1, 0.09269348225),
        (5, 1, 2, 1, 1, 0.01511405958),
        (5, 1, 3, 1, 1, 0.00351323823),
        (5, 1, 1, 2, 1, 0.00986613205),
        (5, 1, 1, 1, 2, 0.01485397749),
        (5, 1, 1, 2, 2, 0.029176944519),
    ],
)
def test_ksg_te_slicing(
    rng_int,
    prop_time,
    step_size,
    src_hist_len,
    dest_hist_len,
    expected,
):
    """Test the KSG transfer entropy estimator with slicing."""
    data_source, data_dest = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    est = KSGTEEstimator(
        data_source,
        data_dest,
        prop_time=prop_time,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        noise_level=0,  # for reproducibility
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,k,ksg_id,minkowski_p,base,expected",
    [
        (5, 4, 1, 2, "e", -0.14824377373692343),
        (5, 4, 2, 2, "e", -0.39824377373692343),
        (5, 4, 1, 3, "e", -0.105711677),
        (5, 4, 1, inf, "e", 0.089420174),
        (5, 16, 1, 2, "e", -0.0629498022),
        (5, 16, 1, 3, "e", -0.017337609),
        (5, 16, 1, inf, "e", 0.082505768),
        (6, 4, 1, 2, "e", -0.1120271189),
        (6, 4, 1, 3, "e", -0.066107282),
        (6, 4, 1, inf, "e", 0.117280006),
        (7, 4, 1, 2, "e", -0.17448870909),
        (7, 4, 1, 3, "e", -0.126082959),
        (7, 4, 1, inf, "e", 0.069872334),
        (7, 4, 1, 2, 2.0, -0.25173399),
        (7, 4, 1, 3, 10, -0.054757133),
        (7, 4, 1, inf, 5.0, 0.043414122),
    ],
)
def test_ksg_cte(rng_int, k, ksg_id, minkowski_p, base, expected):
    """Test the conditional KSG transfer entropy estimator."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        k=k,
        ksg_id=ksg_id,
        minkowski_p=minkowski_p,
        noise_level=0,  # for reproducibility
        base=base,
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,step_size,src_hist_len,dest_hist_len,cond_hist_len,ksg_id,expected",
    [
        (5, 1, 1, 1, 1, 1, 0.089420174),
        (5, 1, 1, 1, 1, 2, -0.2817847788),
        (6, 1, 1, 1, 1, 1, 0.1172800069),
        (6, 1, 1, 1, 1, 2, -0.2554896724),
        (7, 1, 1, 1, 1, 1, 0.0698723348),
        (7, 1, 1, 1, 1, 2, -0.2981837603),
        (5, 1, 1, 1, 1, 1, 0.089420174),
        (5, 2, 1, 1, 1, 1, 0.052874143),
        (5, 3, 1, 1, 1, 1, 0.064791017),
        (5, 1, 2, 1, 1, 1, 0.085618601),
        (5, 1, 1, 2, 1, 1, 0.089773402),
        (5, 1, 1, 1, 2, 1, 0.077874619),
        (5, 1, 2, 2, 1, 1, 0.065095054),
        (5, 1, 2, 2, 2, 1, 0.061523405),
    ],
)
def test_ksg_cte_slicing(
    rng_int,
    step_size,
    src_hist_len,
    dest_hist_len,
    cond_hist_len,
    ksg_id,
    expected,
):
    """Test the conditional KSG transfer entropy estimator with slicing."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        step_size=step_size,
        src_hist_len=src_hist_len,
        dest_hist_len=dest_hist_len,
        cond_hist_len=cond_hist_len,
        ksg_id=ksg_id,
        noise_level=0,  # for reproducibility
    )
    assert isinstance(est.result(), float)
    assert est.result() == pytest.approx(expected)
    assert isinstance(est.local_vals(), ndarray)


@pytest.mark.parametrize(
    "rng_int,method,p_te,p_cte",
    [
        (1, "permutation_test", 0.0, 0.0),
        (1, "bootstrap", 0.0, 0.0),
        (2, "permutation_test", 0.0, 0.0),
        (2, "bootstrap", 0.0, 0.0),
        (3, "permutation_test", 0.0, 0.0),
        (4, "permutation_test", 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("ksg_id", [1, 2])
def test_ksg_te_statistical_test(rng_int, method, p_te, p_cte, ksg_id):
    """Test the KSG TE for p-values. Fix rng."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est_te_xy = KSGTEEstimator(
        data_source,
        data_dest,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base=2,
        seed=8,
        ksg_id=ksg_id,
    )
    est_cte_xy = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_cond,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base=2,
        seed=8,
        ksg_id=ksg_id,
    )
    test = est_te_xy.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_te)
    test = est_cte_xy.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_cte)


@pytest.mark.parametrize(
    "rng_int,method,eff_te_1,eff_te_2,eff_cte_1,eff_cte_2",
    [
        (1, "permutation_test", 0.109132408, 0.102263973, 0.0, 0.0),
        (1, "bootstrap", 0.114234819, 0.108487012, 0.0, 0.0),
        (2, "permutation_test", 0.042919601, 0.044615241, 0.0, 0.0),
        (2, "bootstrap", 0.053020353, 0.054544085, 0.0, 0.0),
        (3, "permutation_test", 0.071168534, 0.067808382, 0.0, 0.0),
        (4, "permutation_test", 0.096478042, 0.089971667, 0.0, 0.0),
    ],
)
def test_ksg_te_effective_val(
    rng_int, method, eff_te_1, eff_te_2, eff_cte_1, eff_cte_2
):
    """Test the KSG transfer entropy for effective values. Fix rng."""
    data_source, data_dest, data_cond = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    est_te_xy_1 = KSGTEEstimator(
        data_source,
        data_dest,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base="e",
        seed=8,
        ksg_id=1,
    )
    est_cte_xy_1 = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_source,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base="e",
        seed=8,
        ksg_id=1,
    )
    assert est_te_xy_1.effective_val(method=method) == pytest.approx(eff_te_1)
    assert est_cte_xy_1.effective_val(method=method) == pytest.approx(eff_cte_1)

    est_te_xy_2 = KSGTEEstimator(
        data_source,
        data_dest,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base="e",
        seed=8,
        ksg_id=2,
    )
    est_cte_xy_2 = KSGCTEEstimator(
        data_source,
        data_dest,
        cond=data_source,
        k=4,
        minkowski_p=inf,
        noise_level=0,
        base="e",
        seed=8,
        ksg_id=2,
    )
    assert est_te_xy_2.effective_val(method=method) == pytest.approx(eff_te_2)
    assert est_cte_xy_2.effective_val(method=method) == pytest.approx(eff_cte_2)


def test_ksg_te_variants():
    """Test KSG TE variants."""
    np.random.seed(42)
    x = np.random.rand(100)
    y = np.roll(x, 1) + np.random.normal(0, 0.1, 100)

    te1 = transfer_entropy(x, y, approach="ksg", k=4, ksg_id=1)
    te2 = transfer_entropy(x, y, approach="ksg", k=4, ksg_id=2)

    assert te1 > 0
    assert te2 > 0
    assert te1 != te2


def test_ksg_invalid_id():
    """Test that invalid ksg_id raises ValueError."""
    x = np.random.rand(10)
    y = np.random.rand(10)

    # TE
    with pytest.raises(ValueError, match="ksg_id must be 1 or 2"):
        KSGTEEstimator(x, y, ksg_id=-1)

    # CTE
    with pytest.raises(ValueError, match="ksg_id must be 1 or 2"):
        KSGCTEEstimator(x, y, cond=x, ksg_id=1.5)
