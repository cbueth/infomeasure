"""Module for the Symbolic / Permutation entropy estimator."""

from collections import Counter

from numpy import array, sum as np_sum

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, PValueMixin


class SymbolicEntropyEstimator(PValueMixin, EntropyEstimator):
    r"""Estimator for the Symbolic / Permutation entropy.

    The Symbolic entropy is a measure of the complexity of a time series.
    The input data needs to be comparable, i.e., the data should be ordinal,
    as the relative frequencies are calculated.
    For a given ``order`` (length of considered subsequences),
    all :math:`n!` possible permutations are considered
    and their relative frequencies are calculated
    :cite:p:`bandtPermutationEntropyNatural2002`.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    order : int
        The size of the permutation patterns.
    per_symbol : bool, optional
        If True, the entropy is divided by the order - 1.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Notes
    -----
    The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.

    Raises
    ------
    ValueError
        If the ``order`` is negative or not an integer.
    ValueError
        If the ``order`` is too large for the given data.

    Warning
    -------
    If ``order`` is set to 1, the entropy is always 0.
    """

    def __init__(
        self,
        data,
        order: int,
        per_symbol: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the SymbolicEntropyEstimator.

        Parameters
        ----------
        order : int
            The order of the Symbolic entropy.
        """
        super().__init__(data, base=base)
        if not isinstance(order, int) or order < 0:
            raise ValueError("The order must be a non-negative integer.")
        if order > len(self.data):
            raise ValueError("The order is too large for the given data.")
        if order == 1:
            logger.warning("The Symbolic entropy is always 0 for order=1.")
        self.order = order
        self.per_symbol = per_symbol

    @staticmethod
    def _estimate_probabilities_order_2(time_series):
        """Simplified case for order 2."""
        gt = time_series[:-1] < time_series[1:]  # compare all neighboring elements
        gt = np_sum(gt) / (len(time_series) - 1)  # sum up the True values
        if gt == 0 or gt == 1:
            return array([1])  # output cannot include zeros
        return array([gt, 1 - gt])  # return the probabilities in the needed format

    @staticmethod
    def _estimate_probabilities_order_3(time_series):
        """Simplified case for order 3."""
        gt1 = time_series[:-2] < time_series[1:-1]  # 0 < 1
        gt2 = time_series[1:-1] < time_series[2:]  # 1 < 2
        gt3 = time_series[:-2] < time_series[2:]  # 0 < 2
        count = Counter(zip(gt1, gt2, gt3))
        probs = array([v / (len(time_series) - 2) for v in count.values()])
        return probs[probs != 0]  # output cannot include zeros

    @staticmethod
    def _get_subarray_patterns(a, n):
        r"""Get the subarray patterns for a given array and order.

        Only sorts the array once and then uses the sorted indices to create the
        patterns.
        This approach is more efficient than the naive approach for these cases:
        Small ascii plot of length against order:
        ```
        length \ order | 2 - 5 | 6 | 7 | 8 | 9 | 12 |
        ---------------------------------------------
        10.000.000     |   X   |   |   |   |   |    |
         1.000.000     |   X   | X |   |   |   |    |
           500.000     |   X   | X | X |   |   |    |
           200.000     |   X   | X | X | X |   |    |
            20.000     |   X   | X | X | X | X |    |
          < 20.000     |   X   | X | X | X | X |  X |
        ```
        Otherwise, the naive approach is faster.
        We do not give the naive approach as alternative, as orders >4 are not to be
        often expected in practice, neither such long time series.
        """
        sorted_indices_a = a.argsort()
        subarray_patterns = [[] for _ in range(len(a))]
        for i in range(len(a)):
            idx = sorted_indices_a[i]
            for i_n in range(n):
                subarray_patterns[idx - i_n].append(i_n)
        return subarray_patterns[: len(a) - n + 1]

    def _estimate_probabilities(self, time_series, order):
        if order == 2:
            return self._estimate_probabilities_order_2(time_series)
        if order == 3:
            return self._estimate_probabilities_order_3(time_series)

        # Get the length of the time series
        total_patterns = len(time_series) - order + 1
        # Create a Counter object to count the occurrences of each permutation
        count = Counter(map(tuple, self._get_subarray_patterns(time_series, order)))
        # Return array of non-zero probabilities
        return array([v / total_patterns for v in count.values()])

    def _calculate(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """

        if self.order == 1:
            return 0.0
        elif self.order == len(self.data):
            return 0.0

        probabilities = self._estimate_probabilities(self.data, self.order)
        # sum over probabilities, multiplied by the logarithm of the probabilities
        entropy = np_sum(-probabilities * self._log_base(probabilities))

        if self.per_symbol:
            entropy /= self.order - 1

        return entropy
