"""Module for Tsallis entropy estimator."""

from ..base import EntropyEstimator, PValueMixin, LogBaseMixin


class TsallisEntropyEstimator(LogBaseMixin, PValueMixin, EntropyEstimator):
    r"""Estimator for the Tsallis entropy.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    q : float
        The Tsallis parameter.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    pass
