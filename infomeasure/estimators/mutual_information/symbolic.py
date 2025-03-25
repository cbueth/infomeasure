"""Module for the Symbolic / Permutation mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer, ndarray

from ..base import (
    PValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..utils.discrete_interaction_information import (
    mutual_information_global,
    mutual_information_local,
    conditional_mutual_information_global,
    conditional_mutual_information_local,
)
from ..utils.symbolic import symbolize_series, reduce_joint_space
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class BaseSymbolicMIEstimator(ABC):
    r"""Base class for the Symbolic mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    order : int
        The size of the permutation patterns.
    stable : bool, optional
        If True, when sorting the data, the order of equal elements is preserved.
        This can be useful for reproducibility and testing, but might be slower.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    *symbols : array-like, shape (n_samples,)
        The symbolized data used to estimate the mutual information.

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
        *data,
        cond=None,
        order: int = None,
        stable: bool = False,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the SymbolicMIEstimator.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        order : int
            The order of the Symbolic entropy.
        stable : bool, optional
            If True, when sorting the data, the order of equal elements is preserved.
            This can be useful for reproducibility and testing, but might be slower.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.
        """
        if cond is None:
            super().__init__(*data, offset=offset, normalize=False, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=False, base=base
            )
        if not issubdtype(type(order), integer) or order < 0:
            raise ValueError("The order must be a non-negative integer.")
        if order == 1:
            logger.warning(
                "The Symbolic mutual information is always 0 for order=1. "
                "Consider using a higher order for more meaningful results."
            )
        self.order = order
        if len(self.data[0]) < (order - 1) + 1:
            raise ValueError("The data is too small for the given order.")
        self.stable = stable

        self.symbols = [
            reduce_joint_space(
                symbolize_series(var, self.order, stable=self.stable, to_int=False)
            )
            for var in self.data
        ]  # Convert permutation tuples to integers for efficiency (reduce_joint_space),
        # so mutual_information_global can use crosstab method internally


class SymbolicMIEstimator(
    BaseSymbolicMIEstimator, PValueMixin, MutualInformationEstimator
):
    r"""Estimator for the Symbolic mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    order : int
        The size of the permutation patterns.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    *symbols : array-like, shape (n_samples,)
        The symbolized data used to estimate the mutual information.

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

        return mutual_information_global(*self.symbols, log_func=self._log_base)

    def _extract_local_values(self) -> ndarray[float]:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of mi.
        """
        return mutual_information_local(*self.symbols, log_func=self._log_base)


class SymbolicCMIEstimator(
    BaseSymbolicMIEstimator, ConditionalMutualInformationEstimator
):
    """Estimator for the Symbolic conditional mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    order : int
        The size of the permutation patterns.
    *symbols : array-like, shape (n_samples,)
        The symbolized data used to estimate the mutual information.
    symbols_cond : array-like, shape (n_samples,)
        The symbolized conditional data used to estimate the
        conditional mutual information.

    Notes
    -----
    The order will be determined via :func:`numpy.argsort() <numpy.argsort>`.
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbols_cond = reduce_joint_space(
            symbolize_series(self.cond, self.order, stable=self.stable, to_int=False)
        )

    def _calculate(self) -> float:
        """Calculate the conditional mutual information of the data."""

        if self.order == 1:
            return 0.0

        return conditional_mutual_information_global(
            *self.symbols,
            cond=self.symbols_cond,
            log_func=self._log_base,
        )

    def _extract_local_values(self) -> ndarray:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of cmi.
        """
        return conditional_mutual_information_local(
            *self.symbols, cond=self.symbols_cond, log_func=self._log_base
        )
