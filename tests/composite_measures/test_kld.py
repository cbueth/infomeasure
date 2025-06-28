"""Tests for the Kulback-Leibler Divergence (KLD) module."""

import pytest

import infomeasure as im
from tests.conftest import (
    generate_autoregressive_series,
    discrete_random_variables,
)


def test_kld_discrete(default_rng):
    """Test the Kulback-Leibler Divergence (KLD) estimator with discrete approach."""
    data_x = default_rng.choice([0, 1, 2, 3, 4, 8, 1, 3, 4], size=1000)
    data_y = default_rng.choice([0, 3, 5, 6, 7, 1, 2, 3, 4], size=1000)

    im.kld(data_x, data_y, approach="discrete")


@pytest.mark.parametrize("embedding_dim", [1, 2, 3, 4, 6])
def test_kld_permutation(default_rng, embedding_dim):
    """Test the Kulback-Leibler Divergence (KLD) estimator with permutation approach."""
    data_x = default_rng.normal(size=(1000))
    data_y = default_rng.normal(size=(1000))
    im.kld(data_x, data_y, approach="permutation", embedding_dim=embedding_dim)


@pytest.mark.parametrize(
    "rng_int,approach,kwargs,expected",
    [
        (1, "discrete", {}, 0.003003442),
        (2, "discrete", {}, 0.0002397373),
        (3, "discrete", {}, 0.005162126),
        (4, "discrete", {}, 0.005725246),
        (1, "miller_madow", {}, 0.0030034420084321667),
        (2, "miller_madow", {}, 0.00023973733136206654),
        (3, "miller_madow", {}, 0.005162126648452281),
        (4, "miller_madow", {}, 0.005725246746834056),
        (1, "mm", {}, 0.0030034420084321667),
        (2, "mm", {}, 0.00023973733136206654),
        (3, "mm", {}, 0.005162126648452281),
        (4, "mm", {}, 0.005725246746834056),
        (1, "permutation", {"embedding_dim": 1}, 0.0),
        (1, "permutation", {"embedding_dim": 2}, -0.01243173),
        (1, "permutation", {"embedding_dim": 3}, -0.03144439),
        (
            1,
            "permutation",
            {"embedding_dim": 4, "stable": True},
            (0.031130479, 0.056176056),
        ),
        (
            1,
            "permutation",
            {"embedding_dim": 5, "stable": True},
            (0.009108079, 0.0570636788),
        ),
        (1, "permutation", {"embedding_dim": 20}, -6.88857245),
        # Bayes tests with varying alpha
        (1, "bayes", {"alpha": 0.5}, 0.0029917680032702787),
        (1, "bayes", {"alpha": 1.0}, 0.0029801619954685776),
        (1, "bayes", {"alpha": 2.0}, 0.002957151866825658),
        (2, "bayes", {"alpha": 0.5}, 0.0002387735390183554),
        (2, "bayes", {"alpha": 1.0}, 0.00023781554775981206),
        (2, "bayes", {"alpha": 2.0}, 0.00023591678308876318),
        (3, "bayes", {"alpha": 0.5}, 0.005141647145739059),
        (3, "bayes", {"alpha": 1.0}, 0.005121289474049773),
        (3, "bayes", {"alpha": 2.0}, 0.005080935769454342),
        (4, "bayes", {"alpha": 0.5}, 0.005702539021492203),
        (4, "bayes", {"alpha": 1.0}, 0.005679966182226881),
        (4, "bayes", {"alpha": 2.0}, 0.00563522090586055),
    ],
)
def test_kld_explicit_discrete(rng_int, approach, kwargs, expected):
    """Test the Kulback-Leibler Divergence (KLD) estimator with explicit values."""
    data_x, data_y = discrete_random_variables(rng_int)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert any(
        im.kld(data_x, data_y, approach=approach, **kwargs) == pytest.approx(e)
        for e in expected
    )


@pytest.mark.parametrize(
    "rng_int,approach,kwargs,expected",
    [
        (5, "kernel", {"bandwidth": 0.01, "kernel": "gaussian"}, 0.42526766),
        (5, "kernel", {"bandwidth": 0.01, "kernel": "box"}, -1.6972311),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "gaussian"}, 0.09027321),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "box"}, -0.1024642),
        (5, "kernel", {"bandwidth": 1, "kernel": "gaussian"}, 0.16238586),
        (5, "kernel", {"bandwidth": 1, "kernel": "box"}, 0.11077146),
        (5, "kl", {"k": 4}, 0.0470393061),
        (5, "kl", {"k": 2}, 0.0420279784),
        (5, "kl", {"k": 5}, 0.0408217178),
        (5, "renyi", {"alpha": 1}, 0.0480398062),
        (5, "renyi", {"alpha": 0.9}, 0.0304874111),
        (5, "renyi", {"alpha": 2}, 0.1552842100),
        (5, "tsallis", {"q": 1}, 0.0480398062),
        (5, "tsallis", {"q": 0.8}, 0.0233009700),
        (5, "tsallis", {"q": 2}, 0.0035068097),
    ],
)
def test_kld_explicit_continuous(rng_int, approach, kwargs, expected):
    """Test the Kulback-Leibler Divergence (KLD) estimator with explicit values."""
    data_x, data_y = generate_autoregressive_series(rng_int, 0.5, 0.6, 0.4)
    assert im.kld(data_x, data_y, approach=approach, **kwargs) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("approach", [None, "unknown"])
def test_kld_invalid_approach(approach):
    """Test the Kulback-Leibler Divergence (KLD) estimator with invalid approach."""
    with pytest.raises(ValueError):
        im.kld([1, 2, 3], [4, 5, 6], approach=approach)
