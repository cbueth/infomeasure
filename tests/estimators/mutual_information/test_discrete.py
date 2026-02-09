"""Explicit discrete mutual information estimator tests."""

import pytest
from numpy import e, log, array

import infomeasure as im
from infomeasure.estimators.mutual_information import (
    DiscreteMIEstimator,
    DiscreteCMIEstimator,
)
from tests.conftest import (
    discrete_random_variables,
    discrete_random_variables_condition,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 10, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], e, log(2)),
        ([1, 0, 1, 0], [1, 0, 1, 0], 4, log(2) / log(4)),
        ([3, 5, 3, 5, 3, 5], [3, 5, 3, 5, 3, 5], 5, log(2) / log(5)),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 8, log(3) / log(8)),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 2.321928094887362),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 10, 0.6989700043360187),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "e", 1.6094379124341003),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 1.0),
        ([1, 1, 0, 0], [0, 0, 1, 1], 2, 1.0),
        ([1, 1, 0, 0], [0, 1, 0, 1], 2, 0.0),
    ],
)
def test_discrete_mi(data_x, data_y, base, expected):
    """Test the discrete mutual information estimator."""
    est = DiscreteMIEstimator(data_x, data_y, base=base)
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


# test with base 2 and different offsets
@pytest.mark.parametrize(
    "data_x,data_y,offset,expected",
    (
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 0, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 1, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 0, 1.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 1, 0.918295),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 0, log(3) / log(2)),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 1, 1.521928),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 2, 1.5),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 3, 1.584962),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 4, 1.0),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 5, 0.0),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 0, 0.646439),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 2, 1.311278),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 5, 1.521928),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 6, 1.0),
    ),
)
def test_discrete_mi_offset(data_x, data_y, offset, expected):
    """Test the discrete mutual information estimator with offset."""
    est = DiscreteMIEstimator(data_x, data_y, offset=offset, base=2)
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize(
    "rng_int_prop,offset,expected",
    [
        ((1, 0), -1, 0.004536015642),
        ((1, 0), 0, 0.00459569422),
        ((1, 0), 1, 1.00271658),
        ((1, 0), 2, 0.005084169045),
        ((2, 0), 0, 0.003345550576),
        ((3, 0), 0, 0.002135943759),
        ((4, 0), 0, 0.0038080445),
        ((5, 0), 0, 0.00447159843),
        ((6, 0), 0, 0.00630947466),
        ((1, 1), 0, 0.0078610830),
        ((1, 1), 1, 0.009452077983),
        ((1, 10), 10, 0.0094047083583),
    ],
)
def test_discrete_mi_explicit(rng_int_prop, offset, expected):
    """Test the discrete mi estimator with explicit data."""
    data = discrete_random_variables(rng_int_prop[0], prop_time=rng_int_prop[1])
    data = tuple(array(d) for d in data)
    est = DiscreteMIEstimator(*data, base=2, offset=offset)
    assert est.global_val() == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize(
    "data,base,expected",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 2, 0.0),
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), 10, 0.0),
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), 2, 2.0),
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), e, 1.386294361),
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), 4, 1.0),
        (
            ([3, 5, 3, 5, 3, 5], [3, 5, 3, 5, 3, 5], [3, 5, 3, 5, 3, 5]),
            5,
            0.86135311,
        ),
        (
            ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]),
            8,
            1.05664166,
        ),
        (
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5]),
            2,
            4.24385618,
        ),
        (
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 2, 4, 5]),
            10,
            1.27752801,
        ),
        (
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 2, 3, 4, 5]),
            "e",
            2.94161695,
        ),
        (([1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]), 2, 2.0),
        (([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]), 2, 1.31127812),
        (([1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1]), 2, 0.81127812),
    ],
)
def test_discrete_mi_3_vars(data, base, expected):
    """Test the discrete mutual information estimator with 3 variables."""
    est = DiscreteMIEstimator(*data, base=base)
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize(
    "rng_int,expected",
    [
        (1, 1.01092021),
        (2, 1.01210455),
        (3, 1.00738208),
        (4, 1.00927966),
        (5, 1.01124761),
        (6, 1.01629579),
        (7, 1.01153139),
    ],
)
def test_discrete_mi_3_vars_explicit(rng_int, expected):
    """Test the discrete mi estimator with explicit data."""
    data = discrete_random_variables_condition(rng_int)
    est = DiscreteMIEstimator(*data, base=2)
    assert est.global_val() == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize("num_vars", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_discrete_mi_n_vars(num_vars, default_rng):
    """Test the discrete mutual information estimator with n variables."""
    data = [default_rng.integers(0, 10, size=1000) for _ in range(num_vars)]
    if num_vars == 1:
        with pytest.raises(ValueError):
            DiscreteMIEstimator(*data, base=2)
        return
    im.mutual_information(*data, approach="discrete")
    est = DiscreteMIEstimator(*data, base=2)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize(
    "data_x,data_y,cond,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 10, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.399999999),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 2, 4, 5], 10, 0.1204119982),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 2, 3, 4, 5], "e", 0.2772588722),
        ([1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], 2, 0.0),
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], 2, 0.688721875),
        ([1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1], 2, 0.18872187554),
    ],
)
def test_discrete_cmi(data_x, data_y, cond, base, expected):
    """Test the discrete conditional mutual information estimator."""
    est = DiscreteCMIEstimator(data_x, data_y, cond=cond, base=base)
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
    est.local_vals()  # Checks internally for `global = mean(local)`


@pytest.mark.parametrize(
    "data,cond,base,expected",
    [
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), [1, 1, 1, 1, 1], 2, 0.0),
        (([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]), [1, 1, 1, 1, 1], 10, 0.0),
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), [1, 0, 1, 0], 2, 1.0),
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), [1, 0, 1, 0], "e", 0.6931471805),
        (([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]), [1, 0, 1, 0], 10, 0.3010299957),
        (([1, 0, 1, 0],) * 4, [1, 0, 1, 0], 2, 2.0),  # 4 times the same data
        (([1, 0, 1, 0],) * 5, [1, 0, 1, 0], 2, 3.0),  # 5 times the same data
        (([1, 0, 1, 0],) * 6, [1, 0, 1, 0], 2, 4.0),  # 6 times the same data
        (([1, 0, 1, 0],) * 2, [1, 0, 1, 0], 2, 0.0),  # 2 times the same data
        (
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5]),
            [1, 2, 3, 1, 5],
            2,
            2.32192809,
        ),
        (
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 2, 4, 5]),
            [1, 2, 2, 4, 5],
            5,
            1.0,
        ),
        (
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 2, 3, 4, 5]),
            [5, 2, 3, 4, 5],
            2,
            2.32192809,
        ),
        (([1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1]), [0, 1, 0, 1], 2, 1.0),
        (([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]), [0, 1, 0, 0], 2, 1.5),
    ],
)
def test_discrete_cmi_multiple_vars(data, cond, base, expected):
    """Test the discrete conditional mutual information estimator with
    multiple variables."""
    est = DiscreteCMIEstimator(*data, cond=cond, base=base)
    res = est.result()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
    est.local_vals()


@pytest.mark.parametrize(
    "cond",
    [
        array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]]),
        (array([0, 0, 0, 0, 1, 1, 1, 1]), array([0, 0, 1, 1, 0, 0, 1, 1])),
    ],
)
def test_discrete_cmi_joint_cond(cond):
    """Test the discrete conditional mutual information with joint condition."""
    data1 = array([0, 1, 0, 1, 0, 1, 0, 1])
    data2 = array([0, 1, 0, 1, 1, 0, 1, 0])

    est = DiscreteCMIEstimator(data1, data2, cond=cond, base="e")
    res = est.result()
    assert res == pytest.approx(log(2))


@pytest.mark.parametrize("rng_int", [1, 2, 3, 4, 5, 6])
def test_entropy_equality(rng_int):
    """Test the equality of MI(x, x) = H(x)."""
    data = discrete_random_variables(rng_int, prop_time=0)
    x = data[0]
    est_mi = DiscreteMIEstimator(x, x, base=2)
    est_ent = im.entropy(data[0], approach="discrete", base=2)
    assert est_mi.result() == pytest.approx(est_ent)


@pytest.mark.parametrize(
    "rng_int,method,p_mi,p_cmi",
    [
        (1, "permutation_test", 0.74, 1.0),
        (1, "bootstrap", 0.82, 1.0),
        (2, "permutation_test", 0.88, 1.0),
        (2, "bootstrap", 0.86, 1.0),
        (3, "permutation_test", 0.98, 1.0),
        (4, "permutation_test", 0.86, 1.0),
    ],
)
def test_discrete_mi_statistical_test(rng_int, method, p_mi, p_cmi):
    """Test the discrete MI for p-values. Fix rng."""
    data_x, data_y, data_cond = discrete_random_variables_condition(rng_int)
    est_mi = DiscreteMIEstimator(data_x, data_y, base=2, seed=8)
    est_cmi = DiscreteCMIEstimator(data_x, data_y, cond=data_x, base=2, seed=8)
    test = est_mi.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_mi)
    test = est_cmi.statistical_test(method=method, n_tests=50)
    assert test.p_value == pytest.approx(p_cmi)
