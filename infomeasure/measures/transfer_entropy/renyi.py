"""Module for the Renyi transfer entropy estimator."""

from ..base import (
    EffectiveTEMixin,
    TransferEntropyEstimator,
)


class RenyiTEEstimator(EffectiveTEMixin, TransferEntropyEstimator):
    r"""Estimator for the Renyi transfer entropy.

    Attributes
    ----------
    source, target : array-like
        The source and target data used to estimate the transfer entropy.
    alpha : float
        The Renyi parameter.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    pass
