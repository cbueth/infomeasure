"""Module for the Symbolic / Permutation transfer entropy estimator."""

from ..base import (
    LogBaseMixin,
    EffectiveTEMixin,
    TransferEntropyEstimator,
)


class SymbolicTEEstimator(LogBaseMixin, EffectiveTEMixin, TransferEntropyEstimator):
    r"""Estimator for the Symbolic transfer entropy.

    Attributes
    ----------
    source, target : array-like
        The source and target data used to estimate the transfer entropy.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    pass
