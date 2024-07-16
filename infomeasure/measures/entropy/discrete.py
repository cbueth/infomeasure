"""Module for the discrete entropy estimator."""

from numpy import add as np_add
from numpy import unique, zeros

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, LogBaseMixin


class DiscreteEntropyEstimator(LogBaseMixin, EntropyEstimator):
    """Estimator for discrete entropy (Shannon entropy).

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Methods
    -------
    calculate()
        Calculate the entropy.
    """

    def __init__(self, data, base: LogBaseType = Config.get("base")):
        """Initialize the DiscreteEntropyEstimator."""
        super().__init__(data, base=base)

    def _calculate(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        # 1. Frequency Counting
        uniq, inverse = unique(self.data, return_inverse=True)
        histogram = zeros(len(uniq), int)  # NaNs are considered as a unique state
        np_add.at(histogram, inverse, 1)
        # 2. Normalization
        histogram = histogram / len(self.data)
        # 3. Entropy Calculation
        return -sum(histogram * self._log_base(histogram))
