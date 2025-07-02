"""Module containing the base classes for the measure estimators."""

import warnings
from abc import ABC, abstractmethod
from io import UnsupportedOperation
from typing import Generic, final, Sequence

from numpy import (
    asarray,
    integer,
    issubdtype,
    log,
    log2,
    log10,
    ndarray,
)
from numpy import mean as np_mean
from numpy import sum as np_sum

from .mixins import StatisticalTestingMixin, EffectiveValueMixin
from .. import Config
from ..utils.config import logger
from ..utils.data import DiscreteData
from ..utils.exceptions import TheoreticalInconsistencyError
from ..utils.types import EstimatorType, LogBaseType
from .utils.normalize import normalize_data_0_1
from .utils.te_slicing import cte_observations, te_observations
from .utils.ordinal import reduce_joint_space


class Estimator(Generic[EstimatorType], ABC):
    """Abstract base class for all measure estimators.

    Find :ref:`Estimator Usage` on how to use the estimators and an overview of the
    available measures (:ref:`Available approaches`).

    Attributes
    ----------
    res_global : float | None
        The global value of the measure.
        None if the measure is not calculated.
    res_local : array-like | None
        The local values of the measure.
        None if the measure is not calculated or if not defined.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    See Also
    --------
    EntropyEstimator, MutualInformationEstimator, TransferEntropyEstimator, DiscreteEntropyEstimator

    Notes
    -----
    The :meth:`_calculate` method needs to be implemented in the derived classes,
    for the local values or the global value.
    From local values, the global value is taken as the mean.
    If is to more efficient to directly calculate the global value,
    it is suggested to have :meth:`_calculate` just return the global value,
    and have the separate :meth:`_extract_local_values` method for the local values,
    which is lazily called by :meth:`local_val`, if needed.
    If the measure has a p-value, the :meth:`p_value` method should be implemented
    (use :class:`StatisticalTestingMixin` for standard implementations).
    """

    def __init__(self, base: LogBaseType = Config.get("base")):
        """Initialize the estimator."""
        self.res_global = None
        self.res_local = None
        self.base = base

    @final
    def calculate(self) -> None:
        """Calculate the measure.

        Estimate the measure and store the results in the attributes.
        """
        results = self._calculate()
        if isinstance(results, ndarray):
            if results.ndim != 1:
                raise RuntimeError(
                    "Local values must be a 1D array. "
                    f"Received {results.ndim}D array with shape {results.shape}."
                )
            self.res_global, self.res_local = np_mean(results), results
            logger.debug(
                f"Global: {self.res_global:.4e}, "
                # show the first max 5 local values
                f"Local: {', '.join([f'{x:.2e}' for x in self.res_local[:5]])}"
                f"{', ...' if len(self.res_local) > 5 else ''}"
            )
        elif isinstance(results, (int, float)):
            self.res_global = results
            logger.debug(f"Global: {self.res_global:.4e}")
        else:
            raise RuntimeError(
                f"Invalid result type {type(results)} for {self.__class__.__name__}."
            )

    @final
    def result(self) -> float:
        """Return the global value of the measure.

        Calculate the measure if not already calculated.

        Returns
        -------
        results : float
           The global value of the measure.
        """
        return self.global_val()

    @final
    def global_val(self) -> float:
        """Return the global value of the measure.

        Calculate the measure if not already calculated.

        Returns
        -------
        global : float
            The global value of the measure.
        """
        if self.res_global is None:
            logger.debug(f"Using {self.__class__.__name__} to estimate the measure.")
            self.calculate()
        return self.res_global

    def local_vals(self):
        """Return the local values of the measure, if available.

        Returns
        -------
        local : array-like
            The local values of the measure.

        Raises
        ------
        io.UnsupportedOperation
            If the local values are not available.
        """
        if self.global_val() is not None and self.res_local is None:
            try:
                self.res_local = self._extract_local_values()
            except NotImplementedError:
                raise UnsupportedOperation(
                    f"Local values are not available for {self.__class__.__name__}."
                )
            # check absolute and relative difference
            if (
                abs(np_mean(self.res_local) - self.res_global) > 1e-10
                and abs((np_mean(self.res_local) - self.res_global) / self.res_global)
                > 1e-5
            ):
                message = (
                    f"Mean of local values {np_mean(self.res_local)} "
                    f"does not match the global value {self.res_global} "
                    f"for {self.__class__.__name__}. "
                    f"Diff: {np_mean(self.res_local) - self.res_global:.2e}. "
                    + (
                        f"As you are using {len(self.data)} random variables, "
                        f"this is likely a numerical error."
                        if (isinstance(self.data, tuple) and len(self.data) > 5)
                        else ""
                    )
                )
                logger.warning(message)
        return self.res_local

    @abstractmethod
    def _calculate(self) -> float | ndarray[float]:
        """Calculate the measure.

        Returns
        -------
        result : float | array-like
            The entropy as float, or an array of local values.
        """
        pass

    def _extract_local_values(self) -> ndarray[float]:
        """Extract the local values of the measure.

        For estimators that only calculate the global value, this method can be
        implemented to extract the local values from the data, e.g. histogram,
        implementation-specific values, etc.

        Returns
        -------
        array-like
            The local values of the measure.
        """
        raise NotImplementedError(
            "Local values are not available for this estimator. "
            "Implement the _extract_local_values method to extract them."
        )

    @final
    def _log_base(self, x):
        """Calculate the logarithm of the data using the specified base.

        Parameters
        ----------
        x : array-like
            The data to calculate the logarithm of.

        Returns
        -------
        array-like
            The logarithm of the data.

        Raises
        ------
        ValueError
            If the logarithm base is negative.

        Notes
        -----
        The logarithm base can be an integer, a float, or "e" for the natural logarithm.
        """
        # Common logarithms
        if self.base == 2:
            return log2(x)
        elif self.base == "e":
            return log(x)
        elif self.base == 10:
            return log10(x)
        # Edge case: log_1(x) = 0
        elif self.base == 0:
            return 0
        # Negative base logarithm is undefined
        elif self.base < 0:
            raise ValueError(f"Logarithm base must be positive, not {self.base}.")
        # General logarithm
        else:
            return log(x) / log(self.base)


class EntropyEstimator(Estimator["EntropyEstimator"], ABC):
    r"""Abstract base class for entropy estimators.

    Estimates simple entropy of a data array or joint entropy of two data arrays.

    Attributes
    ----------
    *data : array-like, shape (n_samples,) or tuple of array-like
        The data used to estimate the entropy.
        When passing a tuple of arrays, the joint entropy is considered.
        When passing two arrays, the cross-entropy is considered,
        the second RV relative to the first RV.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the data is not an array or arrays tuple/list.

    See Also
    --------
    .entropy.discrete.DiscreteEntropyEstimator
    .entropy.kernel.KernelEntropyEstimator
    .entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator
    .entropy.renyi.RenyiEntropyEstimator
    .entropy.ordinal.OrdinalEntropyEstimator
    .entropy.tsallis.TsallisEntropyEstimator

    Notes
    -----
    - Entropy: When passing one array-like object.
    - Joint Entropy: When passing one tuple of array-likes.
    - Cross-Entropy: When passing two array-like objects.
      Then the the second distribution :math:`q`
      is considered relative to the first :math:`p`:

      :math:`-\sum_{i=1}^{n} p_i \log_b q_i`
    """

    def __init__(self, *data, base: LogBaseType = Config.get("base")):
        """Initialize the estimator with the data."""
        # Check valid input data
        if len(data) == 0:
            raise ValueError("Data must be provided.")
        if len(data) > 2:
            raise ValueError(
                "Only one or two array-like objects are allowed. \n"
                "- One data array for normal entropy\n"
                "- Two data arrays for cross-entropy\n"
                "- When given tuples instead of arrays, "
                "they are considered as one, joint RV."
            )
        if len(data) == 1 and not (
            (
                isinstance(data[0], (ndarray, Sequence))
                and not isinstance(data[0], (str, tuple))
            )
            or (
                isinstance(data[0], tuple)
                and all(
                    (
                        isinstance(v, (ndarray, Sequence))
                        and not isinstance(v, (str, tuple))
                    )
                    for v in data[0]
                )
            )
        ):
            raise ValueError(
                "For normal entropy, data must be a single array-like object. "
                "For joint entropy, data must be a tuple of array-like objects. "
                "Pass two separate data for cross-entropy."
            )
        if len(data) == 2 and not all(
            isinstance(var, (ndarray, Sequence)) and not isinstance(var, (str, tuple))
            for var in data
        ):
            raise ValueError(
                "For cross-entropy, data must be two array-like objects. "
                "Tuples for joint variables are not supported. "
                "For (joint) entropy, just pass one argument."
            )
        # Convert to arrays if they are not already
        self.data = tuple(
            asarray(var)
            if not isinstance(var, tuple)
            else tuple(asarray(d) for d in var)
            for var in data
        )
        # differing lengths are allowed for cross-entropy, but not inside joint RVs
        for var in self.data:
            if isinstance(var, tuple) and any(len(d) != len(var[0]) for d in var):
                raise ValueError(
                    "All elements of a joint random variable must have the same length."
                )

        super().__init__(base=base)

    def local_vals(self):
        """Return the local values of the measure, if available.

        For cross-entropy, local values cannot be calculated.

        Returns
        -------
        local : array-like
            The local values of the measure.

        Raises
        ------
        io.UnsupportedOperation
            If the local values are not available.
        """
        # Cross-entropy cannot be calculated locally:
        # if _cross_entropy got overwritten, raise UnsupportedOperation
        if len(self.data) > 1 and "_cross_entropy" in self.__class__.__dict__:
            raise UnsupportedOperation(
                "Local values can only be calculated for (joint) entropy, "
                "not cross-entropy."
            )
        return super().local_vals()

    def _calculate(self) -> float | ndarray[float]:
        """Calculate the entropy of the data.

        Depending on the `data` type, choose simple or joint entropy calculation.

        Returns
        -------
        float | array-like
            The calculated entropy, or local values if available.
        """
        if len(self.data) == 1:
            if isinstance(self.data[0], tuple):
                logger.debug("Calculating joint entropy.")
                return self._joint_entropy()
            logger.debug("Calculating simple entropy.")
            return self._simple_entropy()
        elif len(self.data) == 2:
            logger.debug("Calculating cross-entropy.")
            return self._cross_entropy()
        else:
            raise RuntimeError(
                f"`self.data` has an invalid format (len {len(self.data)})."
            )

    @abstractmethod
    def _simple_entropy(self) -> float | ndarray[float]:
        """Calculate the entropy of one random variable.

        Returns
        -------
        float | array-like
            The calculated entropy, or local values if available.
        """
        pass

    @abstractmethod
    def _joint_entropy(self) -> float | ndarray[float]:
        """Calculate the joint entropy of two random variables.

        Returns
        -------
        float | array-like
            The calculated entropy, or local values if available.
        """
        pass

    def _cross_entropy(self) -> float:
        r"""Calculate the cross-entropy between two distributions.

        .. math::

           H(p, q) = H_{q}(p) = -\sum_{x} p(x) \log q(x)

        Consider self.data[0] as the distribution :math:`p` and self.data[1]
        as the distribution :math:`q`.

        Returns
        -------
        float
            The calculated cross-entropy.

        Notes
        -----
        As cross-entropy is not symmetric,
        data[0] and data[1] are not exchangable.
        Remember this when overriding this method.
        """
        raise NotImplementedError(
            f"Cross-entropy is not implemented for {self.__class__.__name__}."
        )


class MutualInformationEstimator(
    StatisticalTestingMixin, Estimator["MutualInformationEstimator"], ABC
):
    """Abstract base class for mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    offset : int, optional
        If two data arrays are provided:
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from X to Y.
    normalize : bool, optional
        If True, normalize the data before analysis. Default is False.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the data arrays have different lengths.
    ValueError
        If the offset is not an integer.
    ValueError
        If offset is used with more than two data arrays.

    See Also
    --------
    .mutual_information.discrete.DiscreteMIEstimator
    .mutual_information.kernel.KernelMIEstimator
    .mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator
    .mutual_information.renyi.RenyiMIEstimator
    .mutual_information.ordinal.OrdinalMIEstimator
    .mutual_information.tsallis.TsallisMIEstimator
    """

    def __init__(
        self,
        *data,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Initialize the estimator with the data."""
        if len(data) < 2:
            raise ValueError("At least two data arrays are required for MI estimation.")
        if len(data) > 2 and offset not in (0, None):
            raise ValueError("Offset is only supported for two data arrays.")
        self.data: tuple[ndarray] = tuple(asarray(d) for d in data)
        if not all(var.shape[0] == self.data[0].shape[0] for var in self.data):
            raise ValueError(
                "Data arrays must have the same first dimension, "
                f"not {[var.shape[0] for var in self.data]}."
            )
        # Apply the offset
        self.offset = offset
        if self.offset > 0:
            self.data = (
                self.data[0][: -self.offset or None],
                self.data[1][self.offset :],
            )
        elif self.offset < 0:
            self.data = (
                self.data[0][-self.offset :],
                self.data[1][: self.offset or None],
            )
        # Normalize the data
        self.normalize = normalize
        if self.normalize and any(var.ndim != 1 for var in self.data):
            raise ValueError("Data arrays must be 1D for normalization.")
        if self.normalize:
            self.data = tuple(normalize_data_0_1(var) for var in self.data)
        super().__init__(base=base, **kwargs)

    def _generic_mi_from_entropy(
        self,
        estimator: type(EntropyEstimator),
        noise_level: float = 0,
        kwargs: dict = None,
    ) -> float:
        r"""Calculate the mutual information with the entropy estimator.

        Mutual Information (MI) between two random variables :math:`X` and :math:`Y`
        quantifies the amount of information obtained about one variable through the
        other. In terms of entropy (H), MI is expressed as:

        .. math::

                I(X, Y) = H(X) + H(Y) - H(X, Y)

        where :math:`H(X)` is the entropy of :math:`X`, :math:`H(Y)` is the entropy of
        :math:`Y`, and :math:`H(X, Y)` is the joint entropy of :math:`X` and :math:`Y`.
        For an arbitrary number of variables, the formula is:

        .. math::

                I(X_1, X_2, \ldots, X_n) = \sum_{i=1}^n H(X_i) - H(X_1, X_2, \ldots, X_n)

        Parameters
        ----------
        estimator : EntropyEstimator
            The entropy estimator to use.
        noise_level : float, optional
            The standard deviation of the Gaussian noise to add to the data to avoid
            issues with zero distances.
        kwargs : dict
            Additional keyword arguments for the entropy estimator.

        Returns
        -------
        float
            The mutual information between the two variables.

        Notes
        -----
        If possible, estimators should use a dedicated mutual information method.
        This helper method is provided as a generic fallback.
        """
        data = list(var.copy() for var in self.data)

        # Add Gaussian noise to the data if the flag is set
        if noise_level:
            for i_data in range(len(data)):
                data[i_data] = data[i_data].astype(float)
                data[i_data] += self.rng.normal(0, noise_level, data[i_data].shape)

        # Estimators
        estimators = [estimator(var, **kwargs) for var in data]
        estimator_joint = estimator((*data,), **kwargs)
        # return sum(h(x_i)) - h((x_1, x_2, ..., x_n))
        try:
            return (
                np_sum([est.local_vals() for est in estimators], axis=0)
                - estimator_joint.local_vals()
            )
        except (UnsupportedOperation, TheoreticalInconsistencyError):
            return (
                sum([est.global_val() for est in estimators])
                - estimator_joint.global_val()
            )


class ConditionalMutualInformationEstimator(
    StatisticalTestingMixin, Estimator["ConditionalMutualInformationEstimator"], ABC
):
    """Abstract base class for conditional mutual information estimators.

    Conditional Mutual Information (CMI) between two (or more)
    random variables :math:`X` and :math:`Y` given
    a third variable :math:`Z` quantifies the amount of information
    obtained about one variable through the other, conditioned on the third.
    In terms of entropy (H), CMI is expressed as:

    .. math::

            I(X, Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)

    where :math:`H(X, Z)` is the joint entropy of :math:`X` and :math:`Z`,
    :math:`H(Y, Z)` is the joint entropy of :math:`Y` and :math:`Z`,
    :math:`H(X, Y, Z)` is the joint entropy of :math:`X`, :math:`Y`, and :math:`Z`,
    and :math:`H(Z)` is the entropy of :math:`Z`.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    normalize : bool, optional
        If True, normalize the data before analysis. Default is False.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the data arrays have different lengths.
    ValueError
        If the data arrays are not of the same length.
    ValueError
        If normalization is requested for non-1D data.

    See Also
    --------
    .mutual_information.discrete.DiscreteCMIEstimator
    .mutual_information.kernel.KernelCMIEstimator
    .mutual_information.kraskov_stoegbauer_grassberger.KSGCMIEstimator
    .mutual_information.ordinal.OrdinalCMIEstimator
    """

    def __init__(
        self,
        *data,
        cond=None,
        normalize: bool = False,
        offset=None,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Initialize the estimator with the data."""
        if cond is None:
            raise ValueError("Conditional data must be provided for CMI estimation.")
        if offset not in (None, 0):
            raise ValueError("Offset is not supported for CMI estimation.")
        if len(data[0]) != len(data[1]) or len(data[0]) != len(cond):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(data[0])}, {len(data[1])}, and {len(cond)}."
            )
        self.data = tuple(asarray(d) for d in data)
        self.cond = asarray(cond)
        # Normalize the data
        self.normalize = normalize
        if self.normalize and (
            self.data[0].ndim != 1 or self.data[1].ndim != 1 or self.cond.ndim != 1
        ):
            raise ValueError("Data arrays must be 1D for normalization.")
        if self.normalize:
            self.data = tuple(normalize_data_0_1(var) for var in self.data)
            self.cond = normalize_data_0_1(self.cond)
        super().__init__(base=base, **kwargs)

    def _generic_cmi_from_entropy(
        self,
        estimator: type(EntropyEstimator),
        noise_level: float = 0,
        kwargs: dict = None,
    ) -> float:
        """Calculate the conditional mutual information with the entropy estimator.

        Parameters
        ----------
        estimator : EntropyEstimator
            The entropy estimator to use.
        noise_level : float, optional
            The standard deviation of the Gaussian noise to add to the data to avoid
            issues with zero distances.
        kwargs : dict
            Additional keyword arguments for the entropy estimator.

        Returns
        -------
        float
            The conditional mutual information between the variables,
            given the conditional data.

        Notes
        -----
        If possible, estimators should use a dedicated conditional mutual information
        method.
        This helper method is provided as a generic fallback.
        """

        # Ensure source and dest are numpy arrays
        data = list(var.copy() for var in self.data)
        cond = self.cond.copy()

        # Add Gaussian noise to the data if the flag is set
        if noise_level:
            for i_data in range(len(data)):
                data[i_data] = (
                    data[i_data]
                    if data[i_data].dtype == float
                    else data[i_data].astype(float)
                )
                data[i_data] += self.rng.normal(0, noise_level, data[i_data].shape)
            cond = cond if cond.dtype == float else cond.astype(float)
            cond += self.rng.normal(0, noise_level, cond.shape)

        # Make sure that no second noise is in `kwargs`
        if kwargs is not None and "noise_level" in kwargs:
            logger.warning(
                "Do not pass the noise_level as a keyword argument for the estimator, "
                "as it is already handled by the CMI method. Noise level is set to 0. "
                f"Received noise_level={kwargs['noise_level']} when constructing CMI "
                f"with {estimator.__name__}."
            )
            del kwargs["noise_level"]

        # Entropy-based CMI calculation
        if issubclass(estimator, EntropyEstimator):
            est_marginal_cond = [estimator((var, cond), **kwargs) for var in data]
            estimator_joint = estimator((*data, cond), **kwargs)
            est_cond = estimator(cond, **kwargs)
            # return h_x_z + h_y_z - h_x_y_z - h_z
            try:
                return (
                    np_sum([est.local_vals() for est in est_marginal_cond], axis=0)
                    - estimator_joint.local_vals()
                    - est_cond.local_vals()
                )
            except (UnsupportedOperation, TheoreticalInconsistencyError):
                return (
                    sum([est.global_val() for est in est_marginal_cond])
                    - estimator_joint.global_val()
                    - est_cond.global_val()
                )
        else:
            raise ValueError(f"Estimator must be an EntropyEstimator, not {estimator}.")


class TransferEntropyEstimator(
    EffectiveValueMixin,
    StatisticalTestingMixin,
    Estimator["TransferEntropyEstimator"],
    ABC,
):
    """Abstract base class for transfer entropy estimators.

    Attributes
    ----------
    source : array-like, shape (n_samples,)
        The source data used to estimate the transfer entropy (X).
    dest : array-like, shape (n_samples,)
        The destination data used to estimate the transfer entropy (Y).
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Default is no shift.
        Assumed time taken by info to transfer from source to destination.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the data arrays have different lengths.
    ValueError
        If the propagation time is not an integer.

    See Also
    --------
    .transfer_entropy.discrete.DiscreteTEEstimator
    .transfer_entropy.kernel.KernelTEEstimator
    .transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator
    .transfer_entropy.renyi.RenyiTEEstimator
    .transfer_entropy.ordinal.OrdinalTEEstimator
    .transfer_entropy.tsallis.TsallisTEEstimator
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        prop_time: int = 0,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        step_size: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Initialize the estimator with the data."""
        if offset not in (None, 0):
            if prop_time in (None, 0):
                logger.warning(
                    "Using the `offset` parameter as `prop_time`. "
                    "Please use `prop_time` for the propagation time."
                )
                prop_time = offset
            else:
                raise ValueError(
                    "Both `offset` and `prop_time` are set. "
                    "Use only `prop_time` for the propagation time."
                )
        if len(source) != len(dest):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(source)} and {len(dest)}."
            )
        if not issubdtype(type(prop_time), integer):
            raise ValueError(f"Propagation time must be an integer, not {prop_time}.")
        self.source = asarray(source)
        self.dest = asarray(dest)
        # Apply the prop_time
        self.prop_time = prop_time
        if self.prop_time > 0:
            self.source = self.source[: -self.prop_time * step_size or None]
            self.dest = self.dest[self.prop_time * step_size :]
        elif self.prop_time < 0:
            self.source = self.source[-self.prop_time * step_size :]
            self.dest = self.dest[: self.prop_time * step_size or None]
        # Slicing parameters
        self.src_hist_len, self.dest_hist_len = src_hist_len, dest_hist_len
        self.step_size = step_size
        # Permutation/Resample flags - used by the p-value method and te_obs. slicing
        self.permute_src = False
        self.resample_src = False
        # Initialize Estimator ABC with the base
        super().__init__(base=base, **kwargs)

    def _generic_te_from_entropy(
        self,
        estimator: type(EntropyEstimator),
        noise_level: float = 0,
        kwargs: dict = None,
    ):
        r"""Calculate the transfer entropy with the entropy estimator.

        Given the joint processes:
        - :math:`X_{t_n}^{(l)} = (X_{t_n}, X_{t_n-1}, \ldots, X_{t_n-k+1})`
        - :math:`Y_{t_n}^{(k)} = (Y_{t_n}, Y_{t_n-1}, \ldots, Y_{t_n-l+1})`

        The Transfer Entropy from :math:`X` to :math:`Y` can be computed using the
        following formula, which is based on conditional mutual information (MI):

        .. math::

                I(Y_{t_{n+1}}; X_{t_n}^{(l)} | Y_{t_n}^{(k)}) = H(Y_{t_{n+1}} | Y_{t_n}^{(k)}) - H(Y_{t_{n+1}} | X_{t_n}^{(l)}, Y_{t_n}^{(k)})

        Now, we will rewrite the above expression by implementing the chain rule, as:

        .. math::

                I(Y_{t_{n+1}} ; X_{t_n}^{(l)} | Y_{t_n}^{(k)}) = H(Y_{t_{n+1}}, Y_{t_n}^{(k)}) + H(X_{t_n}^{(l)}, Y_{t_n}^{(k)}) - H(Y_{t_{n+1}}, X_{t_n}^{(l)}, Y_{t_n}^{(k)}) - H(Y_{t_n}^{(k)})

        Parameters
        ----------
        estimator : EntropyEstimator
            The entropy estimator to use.
        noise_level : float, optional
            The standard deviation of the Gaussian noise to add to the data to avoid
            issues with zero distances.
        kwargs : dict
            Additional keyword arguments for the entropy estimator.

        Returns
        -------
        float
            The transfer entropy from source to destination.

        Notes
        -----
        If possible, estimators should use a dedicated transfer entropy method.
        This helper method is provided as a generic fallback.
        """

        # Ensure source and dest are numpy arrays
        source = self.source.copy()
        dest = self.dest.copy()

        # If Discrete Estimator and noise_level is set, raise an error
        if estimator.__name__ == "DiscreteEntropyEstimator" and noise_level:
            raise ValueError(
                "Discrete entropy estimator does not support noise_level. "
                "Please use a different estimator."
            )
        # Add Gaussian noise to the data if the flag is set
        if isinstance(noise_level, (int, float)) and noise_level != 0:
            source = source.astype(float)
            dest = dest.astype(float)
            source += self.rng.normal(0, noise_level, source.shape)
            dest += self.rng.normal(0, noise_level, dest.shape)

        (
            joint_space_data,
            dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = te_observations(
            source,
            dest,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            step_size=self.step_size,
            permute_src=self.permute_src,
            resample_src=self.resample_src,
        )

        est_y_history_y_future = estimator(marginal_2_space_data, **kwargs)
        est_x_history_y_history = estimator(marginal_1_space_data, **kwargs)
        est_x_history_y_history_y_future = estimator(joint_space_data, **kwargs)
        est_y_history = estimator(dest_past_embedded, **kwargs)

        # Compute Transfer Entropy
        try:
            return (
                est_y_history_y_future.local_vals()
                + est_x_history_y_history.local_vals()
                - est_x_history_y_history_y_future.local_vals()
                - est_y_history.local_vals()
            )
        except (UnsupportedOperation, TheoreticalInconsistencyError):
            return (
                est_y_history_y_future.global_val()
                + est_x_history_y_history.global_val()
                - est_x_history_y_history_y_future.global_val()
                - est_y_history.global_val()
            )


class ConditionalTransferEntropyEstimator(
    EffectiveValueMixin,
    StatisticalTestingMixin,
    Estimator["ConditionalTransferEntropyEstimator"],
    ABC,
):
    """Abstract base class for conditional transfer entropy estimators.

    Conditional Transfer Entropy (CTE) from source :math:`X` to destination :math:`Y`
    given a condition :math:`Z` quantifies the amount of information obtained about
    the destination variable through the source, conditioned on the condition.

    Attributes
    ----------
    source : array-like, shape (n_samples,)
        The source data used to estimate the transfer entropy (X).
    dest : array-like, shape (n_samples,)
        The destination data used to estimate the transfer entropy (Y).
    cond : array-like, shape (n_samples,)
        The conditional data used to estimate the transfer entropy (Z).
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int
        Number of past observations to consider for the source, destination, and
        conditional data.
    prop_time : int, optional
        Not compatible with the conditional transfer entropy.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        step_size: int = 1,
        prop_time=None,
        offset=None,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Initialize the estimator with the data."""
        if cond is None:
            raise ValueError("Conditional data must be provided for CTE estimation.")
        if len(source) != len(dest) or len(source) != len(cond):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(source)}, {len(dest)}, and {len(cond)}."
            )
        if not issubdtype(type(prop_time), integer):
            raise ValueError(f"Propagation time must be an integer, not {prop_time}.")
        if prop_time not in (None, 0) or offset not in (None, 0):
            raise ValueError(
                "`prop_time`/`offset` are not compatible with the "
                "conditional transfer entropy."
            )
        self.source = asarray(source)
        self.dest = asarray(dest)
        self.cond = asarray(cond)
        # Slicing parameters
        self.src_hist_len = src_hist_len
        self.dest_hist_len = dest_hist_len
        self.cond_hist_len = cond_hist_len
        self.step_size = step_size
        # Initialize Estimator ABC with the base
        super().__init__(base=base, **kwargs)

    def _generic_cte_from_entropy(
        self,
        estimator: type(EntropyEstimator),
        noise_level: float = 0,
        kwargs: dict = None,
    ):
        r"""Calculate the conditional transfer entropy with the entropy estimator.

        Parameters
        ----------
        estimator : EntropyEstimator
            The entropy estimator to use.
        noise_level : float, optional
            The standard deviation of the Gaussian noise to add to the data to avoid
            issues with zero distances.
        kwargs : dict
            Additional keyword arguments for the entropy estimator.

        Returns
        -------
        float
            The conditional transfer entropy from source to destination
            given the condition.

        Notes
        -----
        If possible, estimators should use a dedicated
        conditional transfer entropy method.
        This helper method is provided as a generic fallback.
        """
        # Ensure source, dest, and cond are numpy arrays
        source = self.source.copy()
        dest = self.dest.copy()
        cond = self.cond.copy()

        # Add Gaussian noise to the data if the flag is set
        if isinstance(noise_level, (int, float)) and noise_level != 0:
            source = source.astype(float)
            dest = dest.astype(float)
            cond = cond.astype(float)
            source += self.rng.normal(0, noise_level, source.shape)
            dest += self.rng.normal(0, noise_level, dest.shape)
            cond += self.rng.normal(0, noise_level, cond.shape)

        (
            joint_space_data,
            dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = cte_observations(
            source,
            dest,
            cond,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            cond_hist_len=self.cond_hist_len,
            step_size=self.step_size,
        )

        est_cond_y_history_y_future = estimator(marginal_2_space_data, **kwargs)
        est_x_history_cond_y_history = estimator(marginal_1_space_data, **kwargs)
        est_x_history_cond_y_history_y_future = estimator(joint_space_data, **kwargs)
        est_y_history_cond = estimator(dest_past_embedded, **kwargs)

        # Compute Conditional Transfer Entropy
        try:
            return (
                est_cond_y_history_y_future.local_vals()
                + est_x_history_cond_y_history.local_vals()
                - est_x_history_cond_y_history_y_future.local_vals()
                - est_y_history_cond.local_vals()
            )
        except (UnsupportedOperation, TheoreticalInconsistencyError):
            return (
                est_cond_y_history_y_future.global_val()
                + est_x_history_cond_y_history.global_val()
                - est_x_history_cond_y_history_y_future.global_val()
                - est_y_history_cond.global_val()
            )


class DiscreteHEstimator(EntropyEstimator, ABC):
    """Abstract base class for discrete entropy estimators.

    The `DiscreteHEstimator` class is an abstract base class extending
    :class:`EntropyEstimator`.
    This class is specifically designed to handle entropy estimation for
    discrete variables.
    It ensures that input data is transformed into a format suitable for discrete
    entropy calculations, verifies input data validity, and reduces joint spaces where
    needed.

    It works exclusively with symbolized or discretized data,
    allowing entropy computations to remain accurate and efficient for discrete
    variables. The class
    also manages situations where multiple random variables' joint data can be reduced
    to simplified forms for further statistical analysis. The data, after processing,
    is represented using unique values and counts instead of directly storing the
    original data.

    Attributes
    ----------
    data : tuple[~infomeasure.utils.data.DiscreteData]
        A tuple containing Discrete data objects. Each of them contains,
        ``uniq``, ``counts``, ``N``, ``K``, and the original ``data`` array.
        For normal and joint entropy `len(data) = 1`, for cross-entropy `len(data) = 2`.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs a DiscreteHEstimator object and initializes its data processing
        pipeline for discrete entropy calculation.
        The initializer performs multiple preprocessing steps such as checking the
        integrity of provided data, reducing joint space if applicable,
        and converting input data to a Probability Mass Function (PMF) format.
        This ensures that all subsequent computations are performed on consistent
        and discrete data.

        Parameters
        ----------
        args : tuple
            Positional arguments passed to the parent class constructor.
        kwargs : dict
            Keyword arguments passed to the parent class constructor.

        """
        if isinstance(args[0], DiscreteData):
            self.data = (args[0],)
            self.base = kwargs.get("base", Config.get("base"))
            self.res_global = None
            self.res_local = None
            return

        super().__init__(*args, **kwargs)
        # warn if the data looks like a float array
        self._check_data()
        # reduce any joint space if applicable
        self._reduce_space()
        # Convert to PMF discrete data
        self.data = tuple(DiscreteData.from_data(var) for var in self.data)

    @classmethod
    def from_counts(
        cls, uniq, counts, base: LogBaseType = Config.get("base"), **kwargs
    ):
        """Construct a DiscreteHEstimator from the provided counts.

        DiscreteData validates the data integrity, other validations are skipped.
        This is used for JSD for :class:`DiscreteHEstimator` childs.
        """
        return cls(
            DiscreteData.from_counts(uniq=uniq, counts=counts), base=base, **kwargs
        )

    @classmethod
    def from_probabilities(
        cls,
        uniq,
        probabilities,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Construct a DiscreteHEstimator from the provided probabilities.

        DiscreteData validates the data integrity, other validations are skipped.
        This is used for JSD for :class:`DiscreteHEstimator` childs.
        """
        return cls(
            DiscreteData.from_probabilities(uniq=uniq, probabilities=probabilities),
            base=base,
            **kwargs,
        )

    def _joint_entropy(self):
        raise RuntimeError(
            "Function should not be called, as _simple_entropy should be used after "
            "_reduce_space for the joint entropy calcualtion."
        )

    def _check_data(self):
        """
        Checks the data structure for each variable and verifies whether it is
        properly symbolised or discretised for entropy estimation. Issues a
        warning if the data seems to be in an inappropriate format, such as a
        float array.

        This method ensures that the input data is suitable for performing
        entropy calculations, which may not work correctly with direct float
        values without prior preprocessing.

        Warnings
        --------
        Warning messages are logged if:
        - The corresponding data for a variable in `self.data` is a NumPy array
          of a float type.
        - The corresponding data for a variable is a tuple where any
          marginal distribution is a NumPy array of a float type.

        Notes
        -----
        Designed for discrete Entropy Estimators.

        """
        for i_var in range(len(self.data)):
            if (
                isinstance(self.data[i_var], ndarray)
                and self.data[i_var].dtype.kind == "f"
            ):
                logger.warning(
                    "The data looks like a float array ("
                    f"{self.data[i_var].dtype}). "
                    "Make sure it is properly symbolized or discretized "
                    "for the entropy estimation."
                )
            elif isinstance(self.data[i_var], tuple) and any(
                isinstance(marginal, ndarray) and marginal.dtype.kind == "f"
                for marginal in self.data[i_var]
            ):
                logger.warning(
                    "Some of the data looks like a float array. "
                    "Make sure it is properly symbolized or discretized "
                    "for the entropy estimation."
                )

    def _reduce_space(self):
        """
        Reduces the dimensionality of the space by identifying regions that can be
        collapsed based on the structure of the data.
        Specifically, this method evaluates whether the entries in self.data are
        multidimensional arrays or tuples, and if so, processes them to form a reduced
        joint space representation.
        This method is typically applied to datasets where co-occurrences or unique
        configurations across variables need to be mapped to a simpler or more compact
        representation.

        Notes
        -----
        The discrete Shannon entropy calculations often do not depend on the
        order of the data.
        Consequently, reducing the data to a set of unique integers or enumerated joint
        observations is enough for further statistical processing.
        Multidimensional arrays and tuple entries are handled to enable joint reduction.

        Attributes
        ----------
        data : iterable
            An iterable containing the data to be processed.
            The data can include multidimensional `ndarray` objects or
            tuples representing variable entries. Upon processing, the
            `data` attribute is modified to reflect its reduced form.
        """
        reduce = tuple(
            (isinstance(var, ndarray) and var.ndim > 1) or isinstance(var, tuple)
            for var in self.data
        )
        if any(reduce):
            # As the discrete shannon entropy disregards the order of the data,
            # we can reduce the values to unique integers.
            # In case of having multiple random variables (tuple or list),
            # this enumerates the unique co-occurrences.
            self.data = tuple(
                reduce_joint_space(var) if red else var
                for var, red in zip(self.data, reduce)
            )
