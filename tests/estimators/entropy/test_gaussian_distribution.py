"""Test for the entropy estimators with the analytical solution."""

import pytest
from numpy import pi, e, log as np_log

import infomeasure as im


def analytical_entropy(sigma, base=im.Config.get("base")):
    """Analytical entropy of a normal distributed random variable with standard deviation sigma."""
    h = 0.5 * np_log(2 * pi * e * sigma**2)
    if base == "e":
        return h
    return h / np_log(base)


@pytest.mark.parametrize("sigma", [1, 4, 10])
@pytest.mark.parametrize("base", [2, e, 10])
def test_entropy_gaussian(entropy_approach, sigma, base, default_rng):
    """Test the entropy estimators with Gaussian data.
    Compare this with the analytical entropy of a normal distributed random variable.
    This should work for all approaches except the ordinal entropy estimator, neither
    the Renyi nor the Tsallis entropy estimator (only if their alpha/q is 1).
    """
    approach_str, needed_kwargs = entropy_approach
    data = default_rng.normal(loc=0, scale=sigma, size=1000)
    if approach_str == "discrete":
        data = data.astype(int)
    # if alpha or q in needed_kwargs, set it to 1
    for key in ["alpha", "q"]:
        if key in needed_kwargs:
            needed_kwargs[key] = 1
    needed_kwargs["base"] = base
    est = im.estimator(data, measure="entropy", approach=approach_str, **needed_kwargs)
    if (approach_str in ["ordinal", "symbolic", "permutation"]) or (
        approach_str == "discrete" and sigma < 3
    ):
        assert pytest.approx(est.global_val(), rel=0.1) != analytical_entropy(
            sigma, base
        )
    else:
        assert pytest.approx(est.global_val(), rel=0.1) == analytical_entropy(
            sigma, base
        )
