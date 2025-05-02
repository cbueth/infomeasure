"""Module containing the base classes for the measure estimators."""

from abc import ABC, abstractmethod
from io import UnsupportedOperation
from operator import gt
from typing import Callable, Generic, final, Sequence

from numpy import asarray, integer, issubdtype, log, log2, log10, nan, ndarray, std
from numpy import mean as np_mean
from numpy import sum as np_sum
from numpy.random import default_rng

from .. import Config
from ..utils.config import logger
from ..utils.types import EstimatorType, LogBaseType
from .utils.normalize import normalize_data_0_1
from .utils.te_slicing import cte_observations, te_observations


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
    EntropyEstimator, MutualInformationEstimator, TransferEntropyEstimator

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
    (use :class:`PValueMixin` for standard implementations).
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
                raise RuntimeError(
                    f"Mean of local values {np_mean(self.res_local)} "
                    f"does not match the global value {self.res_global}. "
                    f"Diff: {np_mean(self.res_local) - self.res_global:.2e}. "
                    + (
                        f"As you are using {len(self.data)} random variables, "
                        f"this is likely a numerical error."
                        if (isinstance(self.data, tuple) and len(self.data) > 5)
                        else ""
                    )
                )
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


class RandomGeneratorMixin:
    """Mixin for random state generation.

    Attributes
    ----------
    rng : Generator
        The random state generator.
    """

    def __init__(self, *args, seed=None, **kwargs):
        """Initialize the random state generator."""
        self.rng = default_rng(seed)
        super().__init__(*args, **kwargs)


class MutualInformationEstimator(
    RandomGeneratorMixin, Estimator["MutualInformationEstimator"], ABC
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
        super().__init__(base=base)

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

        # Ensure data are numpy arrays
        data = list(var.astype(float).copy() for var in self.data)

        # Add Gaussian noise to the data if the flag is set
        if noise_level:
            for i_data in range(len(data)):
                data[i_data] += self.rng.normal(0, noise_level, data[i_data].shape)

        # Estimators
        estimators = [estimator(var, **kwargs) for var in data]
        estimator_joint = estimator((*data,), **kwargs)
        # return sum(h(x_i)) - h((x_1, x_2, ..., x_n))
        try:
            return (
                np_sum([est.local_vals() for est in estimators])
                - estimator_joint.local_vals()
            )
        except UnsupportedOperation:
            return (
                sum([est.global_val() for est in estimators])
                - estimator_joint.global_val()
            )


class ConditionalMutualInformationEstimator(
    RandomGeneratorMixin, Estimator["ConditionalMutualInformationEstimator"], ABC
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
        super().__init__(base=base)

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
                (
                    np_sum([est.local_vals() for est in est_marginal_cond])
                    - estimator_joint.local_vals()
                    - est_cond.local_vals()
                )
            except UnsupportedOperation:
                return (
                    sum([est.global_val() for est in est_marginal_cond])
                    - estimator_joint.global_val()
                    - est_cond.global_val()
                )
        else:
            raise ValueError(f"Estimator must be an EntropyEstimator, not {estimator}.")


class TransferEntropyEstimator(
    RandomGeneratorMixin, Estimator["TransferEntropyEstimator"], ABC
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
        super().__init__(base=base)

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
        except UnsupportedOperation:
            return (
                est_y_history_y_future.global_val()
                + est_x_history_y_history.global_val()
                - est_x_history_y_history_y_future.global_val()
                - est_y_history.global_val()
            )


class ConditionalTransferEntropyEstimator(
    RandomGeneratorMixin, Estimator["ConditionalTransferEntropyEstimator"], ABC
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
        super().__init__(base=base)

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
        except UnsupportedOperation:
            return (
                est_cond_y_history_y_future.global_val()
                + est_x_history_cond_y_history.global_val()
                - est_x_history_cond_y_history_y_future.global_val()
                - est_y_history_cond.global_val()
            )


class DistributionMixin:
    """Mixin for Estimators that offer introspection of the distribution.

    Attributes
    ----------
    dist_dict : dict
        The distribution of the data.
        Format: {symbol: probability}
    """

    def __init__(self, *args, **kwargs):
        """Initialize the distribution attribute."""
        self.dist_dict: dict | None = None
        super().__init__(*args, **kwargs)

    def distribution(self) -> dict:
        """Get the distribution of the data.

        Returns
        -------
        dict
            The distribution of the data.
        """
        if self.dist_dict is None:
            self.dist_dict = self._distribution()
        return self.dist_dict

    def _distribution(self) -> dict:
        """Get the distribution of the data.

        Child classes can implement this method to add a dedicated distribution method.
        If not implemented, it's expected that the `calculate` method sets the
        distribution.

        Returns
        -------
        dict
            The distribution of the data.
        """
        if self.dist_dict is None:
            self.calculate()
        if self.dist_dict is None:
            raise UnsupportedOperation(
                "Distribution is not available for this estimator."
            )
        return self.dist_dict


class PValueMixin(RandomGeneratorMixin):
    """Mixin for p-value calculation.

    There are two methods to calculate the p-value:

    - Permutation test: shuffle the data and calculate the measure.
    - Bootstrap: resample the data and calculate the measure.

    The :func:`p_value` can be used to determine a p-value for a measure,
    and :func:`t_score` to get the corresponding t-score.

    To be used as a mixin class with other :class:`Estimator` Estimator classes.
    Inherit before the main class.

    Notes
    -----
    The permutation test is a non-parametric statistical test to determine if the
    observed effect is significant. The null hypothesis is that the measure is
    not different from random, and the p-value is the proportion of permuted
    measures greater than the observed measure.

    Raises
    ------
    NotImplementedError
        If the p-value method is not implemented for the estimator.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the permutation test."""
        self.original_data = None
        self.data = None
        self.p_val: float = None
        self.t_scr: float = None
        self.p_val_method: str = None
        self.n_tests: int = None
        super().__init__(*args, **kwargs)
        if not isinstance(self, (MutualInformationEstimator, TransferEntropyEstimator)):
            raise NotImplementedError(
                "P-value method is not implemented for the estimator."
            )

    def p_value(self, n_tests: int = None, method: str = None) -> float:
        """Calculate the p-value of the measure.

        Method can be "permutation_test" or "bootstrap".

        - Permutation test: shuffle the data and calculate the measure.
        - Bootstrap: resample the data and calculate the measure.

        Parameters
        ----------
        n_tests : int, optional
            Number of permutations or bootstrap samples.
            Needs to be a positive integer.
            Default is None, which means, if :py:meth:`t_score` was calculated before,
            the same number of tests will be used.
        method : str, optional
            The method to calculate the p-value.
            Default is None, which means, if :py:meth:`t_score` was calculated before,
            the same method will be used.
            If :py:meth:`t_score` was not calculated before,
            it will be calculated using the default method set in the configuration.

        Returns
        -------
        p_value : float
            The p-value of the measure.

        Raises
        ------
        ValueError
            If the chosen method is unknown.
        """
        if (
            self.p_val is not None
            and (method == self.p_val_method or method is None)
            and (n_tests == self.n_tests or n_tests is None)
        ):
            return self.p_val
        logger.debug(
            "Calculating the p-value and t-score "
            f"of the measure {self.__class__.__name__} "
            f"using the {method} method with {n_tests} tests."
        )
        self.p_val_method = (
            method if method is not None else Config.get("p_value_method")
        )
        self.n_tests = n_tests
        if isinstance(self, MutualInformationEstimator):
            if len(self.data) != 2:
                raise UnsupportedOperation(
                    "Permutation test on mutual information is only supported "
                    "for two variables."
                )
            method = self.test_mi
        elif isinstance(self, TransferEntropyEstimator):
            method = self.test_te
        else:
            raise NotImplementedError(
                "Permutation test is not implemented for this estimator."
            )
        self.p_val, self.t_scr = method(n_tests)
        return self.p_val

    def t_score(self, n_tests: int = None, method: str = None) -> float:
        """Get the t-score of the measure.

        Parameters
        ----------
        n_tests : int, optional
            Number of permutations or bootstrap samples.
            Needs to be a positive integer.
            Default is None, which means, if :py:meth:`p_value` was calculated before,
            the same number of tests will be used.
        method : str, optional
            The method to calculate the t-score.
            Default is None, which means, if :py:meth:`p_value` was calculated before,
            the same method will be used.
            If :py:meth:`p_value` was not calculated before,
            it will be calculated using the default method set in the configuration.

        Returns
        -------
        t_score : float
            The t-score of the measure.

        Notes
        -----
        The t-score is the difference between the observed value and the mean of the
        permuted values, divided by the standard deviation of the permuted values
        (if it is greater than 0).
        One can get the z-score by converting the t-score using the cumulative
        distribution function (CDF) of the t-distribution and the inverse CDF
        (percent-point function) of the standard normal distribution.

        Raises
        ------
        ValueError
            If the chosen method is unknown.
        """
        if (
            self.t_scr is not None
            and (method == self.p_val_method or method is None)
            and (n_tests == self.n_tests or n_tests is None)
        ):
            return self.t_scr
        self.p_value(n_tests=n_tests, method=method)
        return self.t_scr

    @staticmethod
    def _p_value_t_score(
        observed_value, test_values, comparison: Callable = gt
    ) -> tuple[float, float]:
        """
        Calculate the p-value and t-score of the observed value.
        Given a list of test values, the number of permutations, and the observed value,
        calculate the p-value and t-score.

        Parameters
        ----------
        observed_value : float
            The observed value.
        test_values : array-like
            The test values.
        comparison : operator, optional
            The comparison operator to use.
            Pass `operator.lt` for less than, `operator.gt` for greater than.
            Default is greater.

        Returns
        -------
        float, float
            The p-value and t-score of the measure.

        Raises
        ------
        ValueError
            If the observed value is not a float.
        ValueError
            If the test values are not an array-like.
        ValueError
            If the comparison operator is not a function.
        """
        if not isinstance(observed_value, float):
            raise ValueError("Observed value must be a float.")
        if not isinstance(test_values, (ndarray, list, tuple)):
            raise ValueError("Test values must be an array-like.")
        if not callable(comparison):
            raise ValueError("Comparison operator must be a function.")
        if len(test_values) < 2:
            raise ValueError("Not enough test values for statistical test.")
        test_values = asarray(test_values)

        null_mean = np_mean(test_values)
        null_std = std(test_values, ddof=1)  # Unbiased estimator (dividing by N-1)

        # Compute p-value: proportion of test values greater than the observed value
        #                  (or different operator if specified)
        p_value = np_sum(comparison(test_values, observed_value)) / len(test_values)

        # Compute t-score:
        t_score = (observed_value - null_mean) / null_std if null_std > 0 else nan

        return p_value, t_score

    def _calculate_mi_with_data_selection(self, method_resample_src: Callable):
        """Calculate the measure for the resampled data using specific method."""
        if len(self.original_data) != 2:
            raise ValueError(
                "MI with data selection is only supported for two variables."
            )
        # Shuffle the data
        self.data = (
            method_resample_src(self.original_data[0]),
            self.original_data[1],
        )
        # Calculate the measure
        res_permuted = self._calculate()
        return (
            res_permuted if isinstance(res_permuted, float) else np_mean(res_permuted)
        )

    def test_mi(self, n_tests: int) -> float:
        """Test the mutual information with a permutation test or bootstrap.

        Parameters
        ----------
        n_tests : int
            The number of permutations or bootstrap samples.

        Returns
        -------
        float, float
            The p-value and t-score of the measure.

        Raises
        ------
        ValueError
            If the number of permutations is not a positive integer.
        """
        if not issubdtype(type(n_tests), integer) or n_tests < 1:
            raise ValueError(
                "Number of permutations must be a positive integer, "
                f"not {n_tests} ({type(n_tests)})."
            )
        # Store unshuffled data
        self.original_data = self.data
        # Perform permutations
        if self.p_val_method == "permutation_test":
            method_resample_src = lambda data_src: self.rng.permutation(
                data_src, axis=0
            )
        elif self.p_val_method == "bootstrap":
            method_resample_src = lambda data_src: self.rng.choice(
                data_src, size=data_src.shape[0], replace=True, axis=0
            )
        else:
            raise ValueError(f"Invalid p-value method: {self.p_val_method}.")
        permuted_values = [
            self._calculate_mi_with_data_selection(method_resample_src)
            for _ in range(n_tests)
        ]
        # Restore the original data
        self.data = self.original_data
        return self._p_value_t_score(
            observed_value=self.global_val(), test_values=permuted_values
        )

    def test_te(self, n_tests: int) -> float:
        """Calculate the permutation test for transfer entropy.

        Parameters
        ----------
        n_tests : int
            The number of permutations to perform.

        Returns
        -------
        float
            The p-value of the measure.

        Raises
        ------
        ValueError
            If the number of permutations is not a positive integer.
        """
        if not issubdtype(type(n_tests), integer) or n_tests < 1:
            raise ValueError(
                "Number of permutations must be a positive integer, "
                f"not {n_tests} ({type(n_tests)})."
            )
        # Activate the permutation flag to permute the source data when slicing
        if self.p_val_method == "permutation_test":
            self.permute_src = self.rng
        elif self.p_val_method == "bootstrap":
            self.resample_src = self.rng
        else:
            raise ValueError(f"Invalid p-value method: {self.p_val_method}.")
        permuted_values = [self._calculate() for _ in range(n_tests)]
        if isinstance(permuted_values[0], ndarray):
            permuted_values = [np_mean(x) for x in permuted_values]
        # Deactivate the permutation/resample flag
        self.permute_src, self.resample_src = False, False
        return self._p_value_t_score(
            observed_value=self.global_val(), test_values=permuted_values
        )


class EffectiveValueMixin:
    """Mixin for effective value calculation.

    To be used as a mixin class with :class:`TransferEntropyEstimator` derived classes.
    Inherit before the main class.

    Attributes
    ----------
    res_effective : float | None
        The effective transfer entropy.

    Notes
    -----
    The effective value is the difference between the original
    value and the value calculated for the permuted data.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the estimator with the effective value."""
        self.res_effective = None
        super().__init__(*args, **kwargs)

    def effective_val(self):
        """Return the effective value.

        Calculates the effective value if not already done,
        otherwise returns the stored value.

        Returns
        -------
        effective : float
            The effective value.
        """
        if self.res_effective is None:
            self.res_effective = self._calculate_effective()
        return self.res_effective

    def _calculate_effective(self):
        """Calculate the effective value.

        Returns
        -------
        effective : float
            The effective value.
        """
        # Activate the permutation flag to permute the source data when slicing
        self.permute_src = self.rng
        res_permuted = self._calculate()
        if isinstance(res_permuted, ndarray):
            res_permuted = np_mean(res_permuted)
        # Deactivate the permutation flag
        self.permute_src = False
        # Return difference
        return self.global_val() - res_permuted


class WorkersMixin:
    """Mixin that adds an attribute for the numbers of workers to use.

    Attributes
    ----------
        n_workers : int, optional
            The number of workers to use. Default is 1.
            -1: Use as many workers as CPU cores available.
    """

    def __init__(self, *args, workers=1, **kwargs):
        if workers == -1:
            from multiprocessing import cpu_count

            workers = cpu_count()
        super().__init__(*args, **kwargs)
        self.n_workers = workers
