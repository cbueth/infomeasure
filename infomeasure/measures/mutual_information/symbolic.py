"""Module for the Symbolic / Permutation mutual information estimator."""

from collections import Counter

from numpy import argsort, mean as np_mean

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import PValueMixin, MutualInformationEstimator


class SymbolicMIEstimator(PValueMixin, MutualInformationEstimator):
    r"""Estimator for the Symbolic mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    order : int
        The order of the Symbolic entropy.
    per_symbol : bool, optional
        If True, the entropy is divided by the order - 1.
    step_size : int
        Step size between elements for the ordinal pattern reconstruction.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Notes
    -----
    The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
    There is no ``normalize`` option, as this would not influence the order of the data.

    Raises
    ------
    ValueError
        If the ``order`` is negative or not an integer.
    ValueError
        If the ``order`` is too large for the given data.
    ValueError
        If the ``step_size`` is negative or not an integer.

    Warning
    -------
    If ``order`` is set to 1, the mutual information is always 0.
    """

    def __init__(
        self,
        data_x,
        data_y,
        order: int,
        per_symbol: bool = False,
        step_size: int = 1,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the SymbolicMIEstimator.

        Parameters
        ----------
        order : int
            The order of the Symbolic entropy.
        """
        super().__init__(data_x, data_y, offset=offset, base=base)
        if not isinstance(order, int) or order < 0:
            raise ValueError("The order must be a non-negative integer.")
        if order > len(self.data_x):
            raise ValueError("The order is too large for the given data.")
        if order == 1:
            logger.warning("The Symbolic mutual information is always 0 for order=1.")
        self.order = order
        if not isinstance(step_size, int) or step_size < 0:
            raise ValueError("The step_size must be a non-negative integer.")
        self.step_size = step_size
        self.per_symbol = per_symbol

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        mi : float
            Estimated mutual information between the two datasets.
        local_mi : array
            Local mutual information for each point.
        """

        def _get_pattern_type(subsequence):
            """
            Determine the permutation pattern type of a given subsequence.

            Parameters:
            subsequence (list or array): A subsequence of the time series.

            Returns:
            tuple: A tuple representing the permutation pattern type.
            """
            return tuple(argsort(subsequence))

        def _symbolize_series(series, order, step_size):
            """
            Convert a time series into a sequence of symbols (permutation patterns).

            Parameters:
            series (list or array): The time series to be symbolized.
            order (int): The length of the subsequence to be symbolized.
            step_size (int): The steps between elements in the subsequence.

            Returns:
            list: A list of tuples representing the symbolized series.
            """
            T = len(series)  # Length of the time series
            patterns = []
            for i in range(T - (order - 1) * step_size):  # Iterate over time series
                subsequence = [
                    series[i + j * step_size] for j in range(order)
                ]  # Extract subsequence
                pattern = _get_pattern_type(subsequence)  # Determine pattern type
                patterns.append(pattern)  # Append pattern to list
            return patterns

        def _estimate_probabilities(symbols_x, symbols_y):
            """
            Estimate the joint and marginal probabilities of the symbol sequences.

            Parameters:
            symbols_x (list): Symbolized first time series.
            symbols_y (list): Symbolized second time series.

            Returns:
            dict: Joint probabilities.
            dict: Marginal probabilities for p(x).
            dict: Marginal probabilities for p(y).
            """
            joint_counts = Counter()  # Counter for joint occurrences
            x_counts = Counter()  # Counter for x occurrences
            y_counts = Counter()  # Counter for y occurrences

            for sx, sy in zip(symbols_x, symbols_y):  # Iterate over symbolized series
                joint_pattern = (sx, sy)  # (x_i, y_i)
                x_pattern = sx  # x_i
                y_pattern = sy  # y_i

                # Update counts
                joint_counts[joint_pattern] += 1
                x_counts[x_pattern] += 1
                y_counts[y_pattern] += 1

            # Calculate total counts
            joint_total = sum(joint_counts.values())
            x_total = sum(x_counts.values())
            y_total = sum(y_counts.values())

            # Calculate probabilities
            joint_prob = {k: v / joint_total for k, v in joint_counts.items()}
            x_prob = {k: v / x_total for k, v in x_counts.items()}
            y_prob = {k: v / y_total for k, v in y_counts.items()}

            return joint_prob, x_prob, y_prob

        # Symbolize the time series x and y
        symbols_x = _symbolize_series(self.data_x, self.order, self.step_size)
        symbols_y = _symbolize_series(self.data_y, self.order, self.step_size)

        # Estimate joint and marginal probabilities
        joint_prob, x_prob, y_prob = _estimate_probabilities(symbols_x, symbols_y)

        # Calculate Local Mutual Information
        local_mi = []
        for (sx, sy), p_joint in joint_prob.items():
            p_x = x_prob.get(sx, 0)  # p(x)
            p_y = y_prob.get(sy, 0)  # p(y)

            if p_joint > 0 and p_x > 0 and p_y > 0:
                local_mi_value = self._log_base(p_joint / (p_x * p_y))
                local_mi.extend([local_mi_value] * int(p_joint * len(self.data_x)))

        # Compute average and standard deviation of Local Mutual Information values
        average_mi = np_mean(local_mi)

        return average_mi, local_mi
