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
        (1, "discrete", {}, 1.379779),
        (2, "discrete", {}, 1.38361671),
        (3, "discrete", {}, 1.38066809),
        (4, "discrete", {}, 1.38264482),
        (1, "permutation", {"embedding_dim": 1}, 0.0),
        (1, "permutation", {"embedding_dim": 2}, 0.6349968),
        (1, "permutation", {"embedding_dim": 3}, 1.5890538),
        (
            1,
            "permutation",
            {"embedding_dim": 4, "stable": True},
            (2.58961692, 2.5730411),
        ),  # Apple Silicon: 2.5730411
        (
            1,
            "permutation",
            {"embedding_dim": 5, "stable": True},
            (2.38246712, 2.37888639),
        ),  # Apple Silicon: 2.37888639
        (1, "permutation", {"embedding_dim": 20}, 0.0),
    ],
)
def test_kld_explicit_discrete(rng_int, approach, kwargs, expected):
    """Test the Kulback-Leibler Divergence (KLD) estimator with explicit values."""
    data_x, data_y = discrete_random_variables(rng_int)
    expected = expected if isinstance(expected, tuple) else (expected,)
    print(expected)
    assert any(
        im.kld(data_x, data_y, approach=approach, **kwargs) == pytest.approx(e)
        for e in expected
    )


@pytest.mark.parametrize(
    "rng_int,approach,kwargs,expected",
    [
        (5, "kernel", {"bandwidth": 0.01, "kernel": "gaussian"}, 0.85518464),
        (5, "kernel", {"bandwidth": 0.01, "kernel": "box"}, -4.427867),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "gaussian"}, 3.77301848),
        (5, "kernel", {"bandwidth": 0.1, "kernel": "box"}, -1.2113968),
        (5, "kernel", {"bandwidth": 1, "kernel": "gaussian"}, 4.14695383),
        (5, "kernel", {"bandwidth": 1, "kernel": "box"}, 2.7882814),
        (5, "kl", {"k": 4}, 3.97468156),
        (5, "kl", {"k": 2}, 4.010417125),
        (5, "kl", {"k": 5}, 3.9851807),
        (5, "renyi", {"alpha": 1}, 3.9779022),
        (5, "renyi", {"alpha": 0.9}, 4.00354337),
        (5, "renyi", {"alpha": 2}, 3.748420213),
        (5, "tsallis", {"q": 1}, 3.9779022),
        (5, "tsallis", {"q": 0.8}, 13.49637),
        (5, "tsallis", {"q": 2}, 0.023807621),
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
