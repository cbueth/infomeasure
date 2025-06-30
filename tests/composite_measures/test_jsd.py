"""Tests for the Jensen-Shannon Divergence (JSD) module."""

import pytest

import infomeasure as im
from tests.conftest import (
    generate_autoregressive_series,
    generate_autoregressive_series_condition,
    discrete_random_variables,
    discrete_random_variables_condition,
)


def test_jsd_discrete(default_rng):
    """Test the Jensen-Shannon Divergence (JSD) estimator with discrete approach."""
    data_x = default_rng.choice([0, 1, 2, 3, 4, 8, 1, 3, 4], size=1000)
    data_y = default_rng.choice([0, 3, 5, 6, 7, 1, 2, 3, 4], size=1000)

    im.jsd(data_x, data_y, approach="discrete")


@pytest.mark.parametrize("embedding_dim", [1, 2, 3, 4, 6])
def test_jsd_permutation(default_rng, embedding_dim):
    """Test the Jensen-Shannon Divergence (JSD) estimator with permutation approach."""
    data_x = default_rng.normal(size=(1000))
    data_y = default_rng.normal(size=(1000))
    im.jsd(data_x, data_y, approach="permutation", embedding_dim=embedding_dim)


@pytest.mark.parametrize("bandwidth", [0.1, 1, 10])
@pytest.mark.parametrize("kernel", ["box", "gaussian"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_jsd_kernel(default_rng, bandwidth, kernel, dim):
    """Test the Jensen-Shannon Divergence (JSD) estimator with kernel approach."""
    data_x = default_rng.normal(size=(1000, dim))
    data_y = default_rng.normal(size=(1000, dim))
    im.jsd(data_x, data_y, approach="kernel", bandwidth=bandwidth, kernel=kernel)


@pytest.mark.parametrize(
    "rng_int,approach,kwargs, expected",
    [
        (1, "discrete", {}, 0.0007499220),
        (2, "discrete", {}, 5.99816917e-05),
        (3, "discrete", {}, 0.001288913),
        (4, "discrete", {}, 0.0014298638),
        (1, "permutation", {"embedding_dim": 1}, 0.0),
        (1, "permutation", {"embedding_dim": 2}, 9.0524491e-05),
        (1, "permutation", {"embedding_dim": 3}, 0.00063122072),
        (1, "permutation", {"embedding_dim": 4, "stable": True}, 0.0078668057),
        (1, "permutation", {"embedding_dim": 5, "stable": True}, 0.038191393),
        (1, "permutation", {"embedding_dim": 20}, 0.693147180),
        (1, "permutation", {"embedding_dim": 20, "base": 2}, 1.0),
        (1, "shrink", {}, 0.00021313448129434143),
        (2, "shrink", {}, 0.0),
        (3, "shrink", {}, 0.00042866643860706155),
        (4, "shrink", {}, 6.356074641e-5),
        (1, "bayes", {"alpha": 0.5}, 0.0007470107395435299),
        (1, "bayes", {"alpha": 1.0}, 0.0007441163403578699),
        (1, "bayes", {"alpha": 2.0}, 0.0007383778116207829),
        (2, "bayes", {"alpha": 0.5}, 5.9740474587677284e-05),
        (2, "bayes", {"alpha": 1.0}, 5.950070975990762e-05),
        (2, "bayes", {"alpha": 2.0}, 5.90254905661336e-05),
        (3, "bayes", {"alpha": "jeffrey"}, 0.0012838073319949572),
        (3, "bayes", {"alpha": "laplace"}, 0.001278731243259168),
        (3, "bayes", {"alpha": 2.0}, 0.0012686690753755947),
        (4, "bayes", {"alpha": "jeffrey"}, 0.001424195758761737),
        (4, "bayes", {"alpha": "laplace"}, 0.0014185612955959215),
        (4, "bayes", {"alpha": 2.0}, 0.0014073922583819343),
    ],
)
def test_jsd_explicit_discrete(rng_int, approach, kwargs, expected):
    """Test the Jensen-Shannon Divergence (JSD) estimator with explicit values."""
    data_x, data_y = discrete_random_variables(rng_int)
    assert im.jsd(data_x, data_y, approach=approach, **kwargs) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "rng_int,approach,kwargs, expected",
    [
        (1, "discrete", {}, 0.000717311902),
        (2, "discrete", {}, 0.000123401840),
        (3, "discrete", {}, 0.000906815853),
        (4, "discrete", {}, 0.0013302194),
        (1, "permutation", {"embedding_dim": 1}, 0.0),
        (1, "permutation", {"embedding_dim": 2}, 0.000224832739),
        (1, "permutation", {"embedding_dim": 3}, 0.0011090142),
        (1, "permutation", {"embedding_dim": 4, "stable": True}, 0.0072879551),
        (1, "permutation", {"embedding_dim": 5, "stable": True}, 0.0415068421),
        (1, "permutation", {"embedding_dim": 20}, 1.09861228),
        (1, "permutation", {"embedding_dim": 20, "base": 3}, 1.0),
        (1, "shrink", {}, 0.0002199509867240046),
        (2, "shrink", {}, 0.0),
        (3, "shrink", {}, 0.0002894203847330079),
        (4, "shrink", {}, 7.89759980e-05),
        (1, "bayes", {"alpha": 0.5}, 0.000714526004833349),
        (1, "bayes", {"alpha": 1.0}, 0.0007117563179062092),
        (1, "bayes", {"alpha": 2.0}, 0.0007062650743332455),
        (2, "bayes", {"alpha": 0.5}, 0.00012290985808660615),
        (2, "bayes", {"alpha": 1.0}, 0.00012242081251145898),
        (2, "bayes", {"alpha": 2.0}, 0.00012145143906461264),
        (3, "bayes", {"alpha": "jeffrey"}, 0.0009032255073839757),
        (3, "bayes", {"alpha": "laplace"}, 0.0008996564673597884),
        (3, "bayes", {"alpha": 2.0}, 0.0008925816346696536),
        (4, "bayes", {"alpha": "jeffrey"}, 0.00132492442893839),
        (4, "bayes", {"alpha": "laplace"}, 0.0013196609404377835),
        (4, "bayes", {"alpha": 2.0}, 0.0013092276736734743),
    ],
)
def test_jsd_explicit_discrete_three(rng_int, approach, kwargs, expected):
    """Test the Jensen-Shannon Divergence (JSD) estimator with explicit values."""
    data_x, data_y, data_z = discrete_random_variables_condition(rng_int)
    assert im.jsd(data_x, data_y, data_z, approach=approach, **kwargs) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "rng_int,approach,kwargs, expected",
    [
        (5, "kernel", {"bandwidth": 0.01, "kernel": "gaussian"}, 0.0817015419),
        (5, "kernel", {"bandwidth": 0.01, "kernel": "box"}, 0.56795207),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "gaussian"}, 0.0261682324),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "box"}, 0.17961064),
        (5, "kernel", {"bandwidth": 1, "kernel": "gaussian"}, 0.015830148),
        (5, "kernel", {"bandwidth": 1, "kernel": "box"}, 0.03525284),
    ],
)
def test_jsd_explicit_continuous(rng_int, approach, kwargs, expected):
    """Test the Jensen-Shannon Divergence (JSD) estimator with explicit values."""
    data_x, data_y = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    assert im.jsd(data_x, data_y, approach=approach, **kwargs) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "rng_int,approach,kwargs, expected",
    [
        (5, "kernel", {"bandwidth": 0.01, "kernel": "gaussian"}, 0.11650499),
        (5, "kernel", {"bandwidth": 0.01, "kernel": "box"}, 0.83727519),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "gaussian"}, 0.0381563828),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "box"}, 0.235303841),
        (5, "kernel", {"bandwidth": 1, "kernel": "gaussian"}, 0.0298514962),
        (5, "kernel", {"bandwidth": 1, "kernel": "box"}, 0.051788420),
    ],
)
def test_jsd_explicit_continuous_three(rng_int, approach, kwargs, expected):
    """Test the Jensen-Shannon Divergence (JSD) estimator with explicit values."""
    data_x, data_y, data_z = generate_autoregressive_series_condition(
        rng_int, alpha=(0.5, 0.1), beta=0.6, gamma=(0.4, 0.2)
    )
    assert im.jsd(data_x, data_y, data_z, approach=approach, **kwargs) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("approach", ["renyi", "tsallis", "kl", None, "unknown"])
def test_jsd_unsupported_approach(approach):
    """Test the Jensen-Shannon Divergence (JSD) estimator with unsupported approach."""
    with pytest.raises(ValueError):
        im.jsd([1, 2, 3], [4, 5, 6], approach=approach)
