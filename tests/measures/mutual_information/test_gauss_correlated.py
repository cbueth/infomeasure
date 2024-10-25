"""Test for the mutual information estimators with the analytical solution."""

import pytest
from numpy import log as np_log, corrcoef, log

import infomeasure as im


# Analytical formula for the mutual information of two Gaussian random variables
def mutual_information_gauss(X, Y, base=im.Config.get("base")):
    """Compute the mutual information between two Gaussian random variables.

    Notes
    -----
    ``r`` is the correlation coefficient between X and Y.
    ``I_Gauss`` is the mutual information between X and Y.
    """
    r = corrcoef(X, Y)[0, 1]
    I_Gauss = -0.5 * log(1 - r**2)
    if base == "e":
        return I_Gauss
    return I_Gauss / np_log(base)


def generate_data(N, r, rng):
    cov_matrix = [[10, r], [r, 10]]
    return rng.multivariate_normal([0, 0], cov_matrix, N)


@pytest.mark.parametrize("corr_coeff", [1, 3, 6])
@pytest.mark.parametrize("base", [2, "e", 10])
def test_mi_corellated(mi_approach, corr_coeff, base, default_rng):
    """Test the mutual information estimators with correlated Gaussian data.
    Compare this with the analytical mutual information of two correlated Gaussian
    random variables.
    For Renyi and Tsallis entropy, the analytical solution is not implemented,
    for alpha/q=1 they match the analytical solution.
    """
    approach_str, needed_kwargs = mi_approach
    data = generate_data(1000, corr_coeff, default_rng)
    if approach_str == "discrete":
        data = data.astype(int)
    # if alpha or q in needed_kwargs, set it to 1
    for key in ["alpha", "q"]:
        if key in needed_kwargs:
            needed_kwargs[key] = 1
    if "bandwidth" in needed_kwargs:
        needed_kwargs["bandwidth"] = 3
    if "kernel" in needed_kwargs:
        needed_kwargs["kernel"] = "box"
    needed_kwargs["base"] = base
    est = im.estimator(
        data_x=data[:, 0],
        data_y=data[:, 1],
        measure="mutual_information",
        approach=approach_str,
        **needed_kwargs,
    )
    assert pytest.approx(
        est.global_val(), rel=0.15, abs=0.2
    ) == mutual_information_gauss(data[:, 0], data[:, 1], base=base)
    assert pytest.approx(
        est.effective_val(), rel=0.15, abs=0.2
    ) == mutual_information_gauss(data[:, 0], data[:, 1], base=base)
