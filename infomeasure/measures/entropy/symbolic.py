"""Module for the Symbolic / Permutation entropy estimator."""

from collections import Counter
from itertools import permutations

from numpy import argsort

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
        The order of the Symbolic entropy.
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

    def _calculate(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """

        if self.order == 1:
            return 0.0

        def _get_pattern_type(subsequence):
            return tuple(argsort(subsequence))

        def _estimate_probabilities(time_series, order):
            # Get the length of the time series
            T = len(time_series)
            # Generate all possible permutations of the given order
            permutations_list = list(permutations(range(order)))
            # Create a Counter object to count the occurrences of each permutation
            count = Counter()

            # Lets counts the occurrences of each permutation pattern in the time series
            # using the Counter object.
            # Iterate through the time series with a sliding window of size 'order'
            for i in range(T - order + 1):
                # Get the pattern type (permutation) of the current subsequence
                pattern = _get_pattern_type(time_series[i : i + order])
                count[pattern] += 1  # Increment the count of this pattern

            # Let's create a dictionary where each key is a permutation,
            # and each value is the probability of that permutation
            # (count divided by the total number of patterns).
            # The total number of patterns we have observed
            total_patterns = T - order + 1
            # Calculate the probability of each permutation
            probs = {perm: count[perm] / total_patterns for perm in permutations_list}
            return probs

        def _compute_entropy(probs):
            h = 0
            # loop iterates over the values of the probabilities dictionary
            for p in probs.values():
                if p > 0:
                    h -= p * self._log_base(p)
            return h

        probabilities = _estimate_probabilities(self.data, self.order)
        entropy = _compute_entropy(probabilities)

        if self.per_symbol:
            entropy /= self.order - 1

        return entropy
