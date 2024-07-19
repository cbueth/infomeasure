"""Module for the Symbolic / Permutation entropy estimator."""

from ..base import EntropyEstimator, PValueMixin


class SymbolicEntropyEstimator(PValueMixin, EntropyEstimator):
    r"""Estimator for the Symbolic entropy.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    pass
