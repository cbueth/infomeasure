"""Tests for the Jensen-Shannon Divergence (JSD) module."""

import pytest

import infomeasure as im
from conftest import create_undersampled_data_for_nsb_three
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
        (1, "miller_madow", {}, 1.49924992206),
        (2, "miller_madow", {}, 1.49855998169),
        (3, "miller_madow", {}, 1.49978891374),
        (4, "miller_madow", {}, 1.49992986387),
        (1, "mm", {}, 1.49924992206),
        (2, "mm", {}, 1.49855998169),
        (3, "mm", {}, 1.49978891374),
        (4, "mm", {}, 1.49992986387),
        (1, "permutation", {"embedding_dim": 1}, 0.0),
        (1, "permutation", {"embedding_dim": 2}, 9.0524491e-05),
        (1, "permutation", {"embedding_dim": 3}, 0.00063122072),
        (1, "permutation", {"embedding_dim": 4, "stable": True}, 0.0078668057),
        (1, "permutation", {"embedding_dim": 5, "stable": True}, 0.038191393),
        (1, "permutation", {"embedding_dim": 20}, 0.693147180),
        (1, "permutation", {"embedding_dim": 20, "base": 2}, 1.0),
        (1, "shrink", {}, 0.000606531080453),
        (2, "shrink", {}, -0.000223369073961),
        (3, "shrink", {}, 0.00089651837153),
        (4, "shrink", {}, 0.000127425146879),
        (1, "chao_wang_jost", {}, -1.3851076899311128),
        (2, "chao_wang_jost", {}, -1.3875122611153405),
        (3, "chao_wang_jost", {}, -1.384842418791894),
        (4, "chao_wang_jost", {}, -1.386175716455187),
        (1, "bayes", {"alpha": 0.5}, 0.0024614856242102245),
        (1, "bayes", {"alpha": 1.0}, 0.002588998444477131),
        (1, "bayes", {"alpha": 2.0}, 0.0026217856567916087),
        (2, "bayes", {"alpha": 0.5}, 0.00025742718042209844),
        (2, "bayes", {"alpha": 1.0}, 0.00027216915086891724),
        (2, "bayes", {"alpha": 2.0}, 0.0002761064396641366),
        (3, "bayes", {"alpha": "jeffrey"}, 0.0027549391953760605),
        (3, "bayes", {"alpha": "laplace"}, 0.0028625788385927553),
        (3, "bayes", {"alpha": 2.0}, 0.0028860130053418587),
        (4, "bayes", {"alpha": "jeffrey"}, 0.0015922889135606688),
        (4, "bayes", {"alpha": "laplace"}, 0.0015994567035715335),
        (4, "bayes", {"alpha": 2.0}, 0.0015920734937933112),
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
        (1, "miller_madow", {}, 1.499217311902),
        (2, "miller_madow", {}, 1.49862340184),
        (3, "miller_madow", {}, 1.4994068158),
        (4, "miller_madow", {}, 1.49983021948),
        (1, "mm", {}, 1.499217311902),
        (2, "mm", {}, 1.49862340184),
        (3, "mm", {}, 1.4994068158),
        (4, "mm", {}, 1.49983021948),
        (1, "permutation", {"embedding_dim": 1}, 0.0),
        (1, "permutation", {"embedding_dim": 2}, 0.000224832739),
        (1, "permutation", {"embedding_dim": 3}, 0.0011090142),
        (1, "permutation", {"embedding_dim": 4, "stable": True}, 0.0072879551),
        (1, "permutation", {"embedding_dim": 5, "stable": True}, 0.0415068421),
        (1, "permutation", {"embedding_dim": 20}, 1.09861228),
        (1, "permutation", {"embedding_dim": 20, "base": 3}, 1.0),
        (1, "shrink", {}, 0.0008445583675),
        (2, "shrink", {}, -0.0),
        (3, "shrink", {}, 0.0007514216630),
        (4, "shrink", {}, 0.0001205684214),
        (1, "chao_wang_jost", {}, -1.384792381467528),
        (2, "chao_wang_jost", {}, -1.3874052119360794),
        (3, "chao_wang_jost", {}, -1.384981579281041),
        (4, "chao_wang_jost", {}, -1.3860541773388924),
        (1, "bayes", {"alpha": 0.5}, 0.0027354023473893374),
        (1, "bayes", {"alpha": 1.0}, 0.0028872929198202613),
        (1, "bayes", {"alpha": 2.0}, 0.0029277220600649745),
        (2, "bayes", {"alpha": 0.5}, 0.00035907377554655895),
        (2, "bayes", {"alpha": 1.0}, 0.00037657557263881536),
        (2, "bayes", {"alpha": 2.0}, 0.0003809274090491588),
        (3, "bayes", {"alpha": "jeffrey"}, 0.0025884952914152493),
        (3, "bayes", {"alpha": "laplace"}, 0.0027144615137364436),
        (3, "bayes", {"alpha": 2.0}, 0.0027459806419511956),
        (4, "bayes", {"alpha": "jeffrey"}, 0.001689047107915398),
        (4, "bayes", {"alpha": "laplace"}, 0.0017112515600457012),
        (4, "bayes", {"alpha": 2.0}, 0.0017088778488771883),
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


@pytest.mark.parametrize(
    "rng_int,approach,kwargs, expected",
    [
        (1, "nsb", {}, -1.6837922900495361),
        (2, "nsb", {}, -1.6424894194991995),
        (3, "nsb", {}, -1.5711555083017794),
        (4, "nsb", {}, -1.5899250866053434),
    ],
)
def test_jsd_explicit_discrete_nsb(rng_int, approach, kwargs, expected):
    """Test the Jensen-Shannon Divergence (JSD) estimator with NSB approach
    where output is not NaN."""
    data_x, data_y, _ = create_undersampled_data_for_nsb_three(rng_int)
    assert im.jsd(data_x, data_y, approach=approach, **kwargs) == pytest.approx(
        expected, rel=1e-3
    )


@pytest.mark.parametrize(
    "rng_int,approach,kwargs, expected",
    [
        (1, "nsb", {}, -1.6545764024263367),
        (2, "nsb", {}, -1.639702916617598),
        (3, "nsb", {}, -1.5924556886095116),
        (4, "nsb", {}, -1.593318114039597),
    ],
)
def test_jsd_explicit_discrete_three_nsb(rng_int, approach, kwargs, expected):
    """Test the Jensen-Shannon Divergence (JSD) estimator with NSB approach
    (three variables) where output is not NaN."""
    data_x, data_y, data_z = create_undersampled_data_for_nsb_three(rng_int)
    assert im.jsd(data_x, data_y, data_z, approach=approach, **kwargs) == pytest.approx(
        expected, rel=1e-3
    )


@pytest.mark.parametrize("approach", ["renyi", "tsallis", "kl", None, "unknown"])
def test_jsd_unsupported_approach(approach):
    """Test the Jensen-Shannon Divergence (JSD) estimator with unsupported approach."""
    with pytest.raises(ValueError):
        im.jsd([1, 2, 3], [4, 5, 6], approach=approach)
