"""Module for the Symbolic / Permutation mutual information estimator."""

from ..base import LogBaseMixin, PValueMixin, MutualInformationEstimator


class SymbolicMIEstimator(LogBaseMixin, PValueMixin, MutualInformationEstimator):
    r"""Estimator for the Symbolic mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    pass
