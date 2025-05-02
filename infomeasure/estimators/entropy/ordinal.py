"""Module for the Ordinal / Permutation entropy estimator."""

from collections import Counter

from numpy import (
    array,
    sum as np_sum,
    True_,
    False_,
    integer,
    issubdtype,
    ndarray,
    unique,
    asarray,
)

from ..base import EntropyEstimator, DistributionMixin
from ..utils.ordinal import reduce_joint_space, symbolize_series
from ..utils.unique import unique_vals
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class OrdinalEntropyEstimator(DistributionMixin, EntropyEstimator):
    r"""Estimator for the Ordinal / Permutation entropy.

    The Ordinal entropy is a measure of the complexity of a time series.
    The input data needs to be comparable, i.e., the data should be ordinal,
    as the relative frequencies are calculated.
    For a given ``embedding_dim`` (length of considered subsequences),
    all :math:`n!` possible permutations are considered
    and their relative frequencies are calculated
    :cite:p:`PermutationEntropy2002`.

    Embedding delay is not supported natively.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    embedding_dim : int
        The size of the permutation patterns.
    stable : bool, optional
        If True, when sorting the data, the embedding_dim of equal elements is preserved.
        This can be useful for reproducibility and testing, but might be slower.

    Notes
    -----
    - The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
    - If ``embedding_dim`` is set to 1, the entropy is always 0.

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If the ``embedding_dim`` is too large for the given data.
    TypeError
        If the data are not 1d array-like(s).
    """

    def __init__(
        self,
        *data,
        embedding_dim: int,
        stable: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the OrdinalEntropyEstimator.

        Parameters
        ----------
        embedding_dim : int
            The embedding dimension of the Ordinal entropy.
        stable : bool, optional
            If True, when sorting the data, the order of equal elements is preserved.
            This can be useful for reproducibility and testing, but might be slower.
        """
        super().__init__(*data, base=base)
        for var in self.data:
            if isinstance(var, ndarray) and var.ndim == 1:
                continue
            if isinstance(var, tuple) and all(
                isinstance(var_i, ndarray) and var_i.ndim == 1 for var_i in var
            ):
                continue
            raise TypeError(
                "The data must be a 1D array or tuple of 1D arrays. "
                "Ordinal patterns can only be computed from 1D arrays."
            )
        if not issubdtype(type(embedding_dim), integer) or embedding_dim < 0:
            raise ValueError("The embedding_dim must be a non-negative integer.")
        if (
            isinstance(self.data[0], ndarray) and embedding_dim > self.data[0].shape[0]
        ) or (
            isinstance(self.data[0], tuple) and embedding_dim > self.data[0][0].shape[0]
        ):
            raise ValueError("The embedding_dim is too large for the given data.")
        if embedding_dim == 1:
            logger.warning("The Ordinal entropy is always 0 for embedding_dim=1.")
        self.embedding_dim = embedding_dim
        self.stable = stable
        self.patterns = None

    @staticmethod
    def _estimate_probabilities_embedding_dim_2(time_series):
        """Simplified case for embedding_dim 2."""
        gt = time_series[:-1] < time_series[1:]  # compare all neighboring elements
        # save gt as self.patterns, where 0 -> (1, 0) and 1 -> (0, 1)
        patterns = [(int(not p), int(p)) for p in gt]
        gt = np_sum(gt) / (len(time_series) - 1)  # sum up the True values
        if gt == 0:
            return array([1]), {(1, 0): 1}, patterns
        if gt == 1:
            return array([1]), {(0, 1): 1}, patterns
        return array([gt, 1 - gt]), {(0, 1): gt, (1, 0): 1 - gt}, patterns

    @staticmethod
    def _estimate_probabilities_embedding_dim_3(time_series):
        """Simplified case for embedding_dim 3."""
        gt1 = time_series[:-2] < time_series[1:-1]  # 0 < 1
        gt2 = time_series[1:-1] < time_series[2:]  # 1 < 2
        gt3 = time_series[:-2] < time_series[2:]  # 0 < 2
        count = Counter(zip(gt1, gt2, gt3))
        probs = array([v / (len(time_series) - 2) for v in count.values()])
        # Translate bool keys to numbers (rename keys)
        bool_to_num_map = {
            (True_, True_, True_): (0, 1, 2),
            (True_, False_, True_): (0, 2, 1),
            (False_, True_, True_): (1, 0, 2),
            (True_, False_, False_): (1, 2, 0),
            (False_, True_, False_): (2, 0, 1),
            (False_, False_, False_): (2, 1, 0),
        }
        dist_dict = {
            bool_to_num_map[key]: prob for key, prob in zip(count.keys(), probs)
        }
        patterns = [
            (bool_to_num_map[(p1, p2, p3)]) for p1, p2, p3 in zip(gt1, gt2, gt3)
        ]
        # output cannot include zeros
        return probs[probs != 0], dist_dict, patterns

    @staticmethod
    def _get_subarray_patterns(a, n, stable_argsort=False):
        r"""Get the subarray patterns for a given array and embedding dimension.

        Only sorts the array once and then uses the sorted indices to create the
        patterns.
        This approach is more efficient than the naive approach for these cases:
        Small ascii plot of length against embedding dimenstion:
        ```
        length \ emb_dim | 2 - 5 | 6 | 7 | 8 | 9 | 12 |
        -----------------------------------------------
        10.000.000       |   X   |   |   |   |   |    |
         1.000.000       |   X   | X |   |   |   |    |
           500.000       |   X   | X | X |   |   |    |
           200.000       |   X   | X | X | X |   |    |
            20.000       |   X   | X | X | X | X |    |
          < 20.000       |   X   | X | X | X | X |  X |
        ```
        Otherwise, the naive approach is faster.
        We do not give the naive approach as alternative,
        as embedding dimensions >4 are not to be often expected in practice,
        neither such long time series.
        """
        sorted_indices_a = a.argsort(stable=stable_argsort)
        subarray_patterns = [[] for _ in range(len(a))]
        for i in range(len(a)):
            idx = sorted_indices_a[i]
            for i_n in range(n):
                subarray_patterns[idx - i_n].append(i_n)
        return subarray_patterns[: len(a) - n + 1]

    def _estimate_probabilities(self, time_series, embedding_dim):
        if embedding_dim == 2:
            return self._estimate_probabilities_embedding_dim_2(time_series)
        if embedding_dim == 3:
            return self._estimate_probabilities_embedding_dim_3(time_series)

        # Get the length of the time series
        total_patterns = len(time_series) - embedding_dim + 1
        # Save the symbolized data
        patterns = self._get_subarray_patterns(
            time_series, embedding_dim, stable_argsort=self.stable
        )
        # Create a Counter object to count the occurrences of each permutation
        count = Counter(map(tuple, patterns))
        # Return array of non-zero probabilities
        probs = array([v / total_patterns for v in count.values()])
        dist_dict = dict(zip(count.keys(), probs))
        return probs, dist_dict, patterns

    def _simple_entropy(self) -> float:
        """Calculate the entropy of the data."""
        if self.embedding_dim == 1:
            self.dist_dict = {0: 1.0}
            self.patterns = [0] * len(self.data[0])
            return 0.0
        elif self.embedding_dim == self.data[0].shape[0]:
            self.dist_dict = {tuple(self.data[0].argsort()): 1.0}
            self.patterns = [list(self.dist_dict.keys())[0]]
            return 0.0
        probabilities, self.dist_dict, self.patterns = self._estimate_probabilities(
            self.data[0], self.embedding_dim
        )
        # sum over probabilities, multiplied by the logarithm of the probabilities
        # we do not return these 'local' values, as these are not local to the input
        # data, but local in relation to the permutation patterns, so the identity
        # used in the Estimator parent class does not work here
        return -np_sum(probabilities * self._log_base(probabilities))

    def _joint_entropy(self) -> float:
        """Calculate the joint entropy of the data."""
        # Symbolize separately (permutation patterns -> Lehmer codes)
        symbols = (
            symbolize_series(marginal, self.embedding_dim, to_int=True)
            for marginal in self.data[0]  # data is tuple of time series
        )  # shape (n - (embedding_dim - 1), num_joints)
        # Reduce the joint space
        self.patterns = reduce_joint_space(
            symbols
        )  # reduction columns stacks the symbols
        # Calculate frequencies of co-ocurrent patterns
        uniq, counts, self.dist_dict = unique_vals(self.patterns)
        probabilities = asarray(list(self.dist_dict.values()))
        # Calculate the entropy
        return -np_sum(probabilities * self._log_base(probabilities))

    def _extract_local_values(self):
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of entropy.
        """
        # Use the saved patterns to calculate the local values
        p_local = [self.dist_dict[tuple(pattern)] for pattern in self.patterns]
        return -self._log_base(p_local)

    def _cross_entropy(self) -> float:
        """Calculate the ordinal cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        if any(isinstance(var, tuple) for var in self.data):
            raise ValueError(
                "Cross-entropy only accepts 1D data, no joint data. "
                "This is to have a clear definition for the support set."
            )
        symbols = tuple(
            symbolize_series(var, self.embedding_dim, to_int=True) for var in self.data
        )
        uniq_p, counts_p, dist_p = unique_vals(symbols[0])
        uniq_q, counts_q, dist_q = unique_vals(symbols[1])
        # Only consider the values where both RV have the same support
        uniq = list(set(uniq_p).intersection(set(uniq_q)))  # P ∩ Q
        if len(uniq) == 0:
            logger.warning("No common support between the two distributions.")
            return 0.0
        return -np_sum([dist_p[val] * self._log_base(dist_q[val]) for val in uniq])
