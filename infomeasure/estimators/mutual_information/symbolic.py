"""Module for the Symbolic / Permutation mutual information estimator."""

from abc import ABC
from collections import Counter

from numpy import sum as np_sum, issubdtype, integer

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..utils.symbolic import symbolize_series


class BaseSymbolicMIEstimator(ABC):
    r"""Base class for the Symbolic mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the (conditional) mutual information.
    data_z : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    order : int
        The size of the permutation patterns.
    stable : bool, optional
        If True, when sorting the data, the order of equal elements is preserved.
        This can be useful for reproducibility and testing, but might be slower.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``data_z`` parameter / conditional MI.

    Notes
    -----
    The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
    There is no ``normalize`` option, as this would not influence the order of the data.

    Raises
    ------
    ValueError
        If the ``order`` is negative or not an integer.
    ValueError
        If ``offset`` and ``order`` are such that the data is too small.

    Warning
    -------
    If ``order`` is set to 1, the mutual information is always 0.
    """

    def __init__(
        self,
        data_x,
        data_y,
        data_z=None,
        *,  # all following parameters are keyword-only
        order: int = None,
        stable: bool = False,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the SymbolicMIEstimator.

        Parameters
        ----------
        data_x, data_y : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
        data_z : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        order : int
            The order of the Symbolic entropy.
        stable : bool, optional
            If True, when sorting the data, the order of equal elements is preserved.
            This can be useful for reproducibility and testing, but might be slower.
        """
        if data_z is None:
            super().__init__(data_x, data_y, offset=offset, base=base)
        else:
            super().__init__(data_x, data_y, data_z, offset=offset, base=base)
        if not issubdtype(type(order), integer) or order < 0:
            raise ValueError("The order must be a non-negative integer.")
        if order == 1:
            logger.warning("The Symbolic mutual information is always 0 for order=1.")
        self.order = order
        if len(self.data_x) < (order - 1) + 1:
            raise ValueError("The data is too small for the given order.")
        self.stable = stable


class SymbolicMIEstimator(
    BaseSymbolicMIEstimator, EffectiveValueMixin, MutualInformationEstimator
):
    r"""Estimator for the Symbolic mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    order : int
        The size of the permutation patterns.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
    There is no ``normalize`` option, as this would not influence the order of the data.

    Raises
    ------
    ValueError
        If the ``order`` is negative or not an integer.
    ValueError
        If ``offset`` and ``order`` are such that the data is too small.

    Warning
    -------
    If ``order`` is set to 1, the mutual information is always 0.
    """

    def _calculate(self) -> float:
        """Calculate the mutual information of the data."""

        if self.order == 1:
            return 0.0

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
                sx, sy = tuple(sx), tuple(sy)  # Convert to tuples
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
        symbols_x = symbolize_series(self.data_x, self.order, stable=self.stable)
        symbols_y = symbolize_series(self.data_y, self.order, stable=self.stable)

        # Estimate joint and marginal probabilities
        joint_prob, x_prob, y_prob = _estimate_probabilities(symbols_x, symbols_y)

        # Calculate Mutual Information for each pattern
        mi_perm = []
        for (sx, sy), p_joint in joint_prob.items():
            p_x = x_prob.get(sx, 0)  # p(x)
            p_y = y_prob.get(sy, 0)  # p(y)

            if p_joint > 0 and p_x > 0 and p_y > 0:  # can be assured due to the counter
                mi_perm.append(p_joint * self._log_base(p_joint / (p_x * p_y)))
        if len(mi_perm) == 0:
            return 0.0

        # # Compute average mutual information for the permutation coincidences
        # # we do not return these 'local' values, as these are not local to the input
        # # data, but local in relation to the permutation patterns, so the identity
        # # used in the Estimator parent class does not work here
        # # it could be done like this:
        # symbols_x_prob = asarray([x_prob.get(tuple(sx), 0) for sx in symbols_x])
        # symbols_y_prob = asarray([y_prob.get(tuple(sy), 0) for sy in symbols_y])
        # joint_prob = asarray(
        #     [joint_prob[(tuple(sx), tuple(sy))] for sx, sy in zip(symbols_x, symbols_y)]
        # )
        #
        # print(f"len(symbols_x_prob): {len(symbols_x_prob)}")
        # return self._log_base(joint_prob / (symbols_x_prob * symbols_y_prob))

        return np_sum(mi_perm)


class SymbolicCMIEstimator(
    BaseSymbolicMIEstimator, ConditionalMutualInformationEstimator
):
    """Estimator for the Symbolic conditional mutual information.

    Attributes
    ----------
    data_x, data_y, data_z : array-like
        The data used to estimate the conditional mutual information.
    order : int
        The size of the permutation patterns.

    Notes
    -----
    The ordinality will be determined via :func:`numpy.argsort() <numpy.argsort>`.
    There is no ``normalize`` option, as this would not influence the order of the data.

    Raises
    ------
    ValueError
        If the ``order`` is negative or not an integer.
    ValueError
        If ``offset`` and ``order`` are such that the data is too small.

    Warning
    -------
    If ``order`` is set to 1, the mutual information is always 0.
    """

    def _calculate(self) -> float:
        """Calculate the conditional mutual information of the data."""

        if self.order == 1:
            return 0.0

        def _estimate_probabilities(symbols_x, symbols_y, symbols_z):
            xyz_counts = Counter()
            xz_counts = Counter()
            yz_counts = Counter()
            z_counts = Counter()

            for sx, sy, sz in zip(symbols_x, symbols_y, symbols_z):
                sx, sy, sz = tuple(sx), tuple(sy), tuple(sz)  # Convert to tuples
                xyz_counts[(sx, sy, sz)] += 1
                xz_counts[(sx, sz)] += 1
                yz_counts[(sy, sz)] += 1
                z_counts[sz] += 1

            # Calculate total counts
            xyz_total = sum(xyz_counts.values())
            xz_total = sum(xz_counts.values())
            yz_total = sum(yz_counts.values())
            z_total = sum(z_counts.values())

            # Calculate probabilities
            xyz_prob = {k: v / xyz_total for k, v in xyz_counts.items()}
            xz_prob = {k: v / xz_total for k, v in xz_counts.items()}
            yz_prob = {k: v / yz_total for k, v in yz_counts.items()}
            z_prob = {k: v / z_total for k, v in z_counts.items()}

            return xyz_prob, xz_prob, yz_prob, z_prob

        # Symbolize the time series
        symbols_x = symbolize_series(self.data_x, self.order, stable=self.stable)
        symbols_y = symbolize_series(self.data_y, self.order, stable=self.stable)
        symbols_z = symbolize_series(self.data_z, self.order, stable=self.stable)

        # Estimate joint and marginal probabilities
        # joint_prob, x_prob, y_prob = _estimate_probabilities(symbols_x, symbols_y)
        xyz_prob, xz_prob, yz_prob, z_prob = _estimate_probabilities(
            symbols_x, symbols_y, symbols_z
        )

        cmi_perm = []
        for (sx, sy, sz), p_xyz in xyz_prob.items():
            p_xz = xz_prob.get((sx, sz), 0)
            p_yz = yz_prob.get((sy, sz), 0)
            p_z = z_prob.get(sz, 0)

            if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                cmi_perm.append(p_xyz * self._log_base(p_xyz * p_z / (p_xz * p_yz)))
        if len(cmi_perm) == 0:
            return 0.0

        return np_sum(cmi_perm)
