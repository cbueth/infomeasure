"""Module containing the base classes for the measure estimators."""

from abc import ABC, abstractmethod
from io import UnsupportedOperation
from typing import final

from numpy import std as np_std, asarray
from numpy import array, log, log2, log10, ndarray
from numpy import sum as np_sum
from numpy import mean as np_mean
from numpy.random import default_rng

from .. import Config
from ..utils.config import logger
from ..utils.types import LogBaseType
from .utils.normalize import normalize_data_0_1
from .utils.te_slicing import te_observations


class Estimator(ABC):
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
    res_std : float | None
        The standard deviation of the local values.
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
    The :meth:`_calculate` method needs to be implemented in the derived classes.
    If the measure has a p-value, the :meth:`p_value` method should be implemented.
    """

    def __init__(self, base: LogBaseType = Config.get("base")):
        """Initialize the estimator."""
        self.res_global = None
        self.res_local = None
        self.res_std = None
        self.base = base

    @final
    def calculate(self):
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
            # TODO: better nanmean?
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

    def results(self):
        """Return the (global, local, std) if available.

        Calculate the measure if not already calculated.

        Returns
        -------
        results : tuple | float
            Tuple of the global, local, and standard deviation values,
            or just the global value if others are not available.

        Notes
        -----
        The local and standard deviation values are not available for all measures.
        """
        self.global_val()
        try:
            return self.res_global, self.local_val(), self.std_val()
        except UnsupportedOperation:
            return self.res_global

    @final
    def global_val(self):
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

    @final
    def local_val(self):
        """Return the local values of the measure, if available.

        Returns
        -------
        local : array-like
            The local values of the measure.

        Raises
        ------
        io.UnsupportedOperation
            If the local values are not available.

        Notes
        -----
        Not available for :class:`EntropyEstimator` classes.
        """
        if self.global_val() is not None and self.res_local is None:
            raise UnsupportedOperation(
                f"Local values are not available for {self.__class__.__name__}."
            )
        return self.res_local

    def std_val(self):
        """Return the standard deviation of the local values, if available.

        Returns
        -------
        std : float
            The standard deviation of the local values.

        Raises
        ------
        io.UnsupportedOperation
            If the standard deviation is not available.

        Notes
        -----
        Not available for :class:`EntropyEstimator` classes.
        """
        if self.res_std is None:
            try:
                self.res_std = np_std(self.local_val())
            except UnsupportedOperation:
                raise UnsupportedOperation(
                    "Standard deviation is not available for the measure "
                    f"{self.__class__.__name__}, as local values are not available."
                )
        return self.res_std

    @abstractmethod
    def _calculate(self) -> tuple | float:
        """Calculate the measure.

        Returns
        -------
        result : float | array-like
            The entropy as float, or an array of local values.
        """
        pass

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


class EntropyEstimator(Estimator, ABC):
    """Abstract base class for entropy estimators.

    Estimates simple entropy of a data array or joint entropy of two data arrays.

    Attributes
    ----------
    data : array-like, shape (n_samples,) or tuple of array-like
        The data used to estimate the entropy.
        When passing tuple of arrays, the joint entropy is considered.
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
    .entropy.symbolic.SymbolicEntropyEstimator
    .entropy.tsallis.TsallisEntropyEstimator
    """

    def __init__(self, data, base: LogBaseType = Config.get("base")):
        """Initialize the estimator with the data."""
        if isinstance(data, tuple):
            if all(isinstance(d, (ndarray, list)) for d in data):
                self.data = tuple(asarray(d) for d in data)
            else:
                raise ValueError(
                    "Data in the tuple must be arrays, not "
                    f"{[type(d) for d in data if not isinstance(d, ndarray)]}."
                )
        else:
            self.data = asarray(data)
        super().__init__(base=base)

    def _calculate(self) -> tuple | float:
        """Calculate the entropy of the data.

        Depending on the `data` type, chooses simple or joint entropy calculation.

        Returns
        -------
        float | array-like
            The calculated entropy, or local values if available.
        """
        if isinstance(self.data, tuple):
            return self._joint_entropy()
        return self._simple_entropy()

    @abstractmethod
    def _simple_entropy(self) -> tuple | float:
        """Calculate the entropy of one random variable.

        Returns
        -------
        float | array-like
            The calculated entropy, or local values if available.
        """
        pass

    @abstractmethod
    def _joint_entropy(self) -> tuple | float:
        """Calculate the joint entropy of two random variables.

        Returns
        -------
        float | array-like
            The calculated entropy, or local values if available.
        """
        pass


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


class MutualInformationEstimator(RandomGeneratorMixin, Estimator, ABC):
    """Abstract base class for mutual information estimators.

    Attributes
    ----------
    data_x, data_y : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
    offset : int, optional
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
        If the data arrays are not of the same length.

    See Also
    --------
    .mutual_information.discrete.DiscreteMIEstimator
    .mutual_information.kernel.KernelMIEstimator
    .mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator
    .mutual_information.renyi.RenyiMIEstimator
    .mutual_information.symbolic.SymbolicMIEstimator
    .mutual_information.tsallis.TsallisMIEstimator
    """

    def __init__(
        self,
        data_x,
        data_y,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data."""
        offset = offset or 0  # Ensure offset is an integer (`None` -> 0)
        if not isinstance(offset, int):
            raise ValueError(f"Offset must be an integer, not {offset}.")
        self.data_x = asarray(data_x)
        self.data_y = asarray(data_y)
        if self.data_x.shape[0] != self.data_y.shape[0]:
            raise ValueError(
                "Data arrays must have the same first dimension, "
                f"not {self.data_x.shape[0]} and {self.data_y.shape[0]}."
            )
        # Apply the offset
        self.offset = offset
        if self.offset > 0:
            self.data_x = self.data_x[: -self.offset or None]
            self.data_y = self.data_y[self.offset :]
        elif self.offset < 0:
            self.data_x = self.data_x[-self.offset :]
            self.data_y = self.data_y[: self.offset or None]
        # Normalize the data
        self.normalize = normalize
        if self.normalize and (self.data_x.ndim != 1 or self.data_y.ndim != 1):
            raise ValueError("Data arrays must be 1D for normalization.")
        if self.normalize:
            self.data_x = normalize_data_0_1(self.data_x)
            self.data_y = normalize_data_0_1(self.data_y)
        super().__init__(base=base)

    def _generic_mi_from_entropy(
        self,
        estimator: type(EntropyEstimator),
        noise_level: float = 0,
        kwargs: dict = None,
    ) -> float:
        """Calculate the mutual information with the entropy estimator.

        Mutual Information (MI) between two random variables :math:`X` and :math:`Y`
        quantifies the amount of information obtained about one variable through the
        other. In terms of entropy (H), MI is expressed as:

        .. math::

                I(X, Y) = H(X) + H(Y) - H(X, Y)

        where :math:`H(X)` is the entropy of :math:`X`, :math:`H(Y)` is the entropy of
        :math:`Y`, and :math:`H(X, Y)` is the joint entropy of :math:`X` and :math:`Y`.

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

        # Ensure source and dest are numpy arrays
        data_x = self.data_x.astype(float).copy()
        data_y = self.data_y.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if noise_level:
            data_x += self.rng.normal(0, noise_level, data_x.shape)
            data_y += self.rng.normal(0, noise_level, data_y.shape)

        h_x = estimator(self.data_x, **kwargs).global_val()
        h_y = estimator(self.data_y, **kwargs).global_val()
        h_xy = estimator((self.data_x, self.data_y), **kwargs).global_val()
        return h_x + h_y - h_xy


class ConditionalMutualInformationEstimator(RandomGeneratorMixin, Estimator, ABC):
    """Abstract base class for conditional mutual information estimators.

    Attributes
    ----------
    data_x, data_y, data_z : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
    normalize : bool, optional
        If True, normalize the data before analysis. Default is False.
    offset : None
        Not compatible with the conditional mutual information.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the data arrays have different lengths.
    ValueError
        If an offset is provided.
    ValueError
        If the data arrays are not of the same length.
    ValueError
        If normalization is requested for non-1D data.

    See Also
    --------
    .mutual_information.discrete.DiscreteCMIEstimator
    .mutual_information.kernel.KernelCMIEstimator
    .mutual_information.kraskov_stoegbauer_grassberger.KSGCMIEstimator
    .mutual_information.symbolic.SymbolicCMIEstimator
    """

    def __init__(
        self,
        data_x,
        data_y,
        data_z,
        normalize: bool = False,
        offset=None,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data."""
        if len(data_x) != len(data_y) or len(data_x) != len(data_z):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(data_x)}, {len(data_y)}, and {len(data_z)}."
            )
        if offset not in (None, 0):
            raise ValueError(
                "`offset` is not compatible with the conditional mutual information."
            )
        self.data_x = asarray(data_x)
        self.data_y = asarray(data_y)
        self.data_z = asarray(data_z)
        if (
            self.data_x.shape[0] != self.data_y.shape[0]
            or self.data_x.shape[0] != self.data_z.shape[0]
        ):
            raise ValueError(
                "Data arrays must have the same first dimension, "
                f"not {self.data_x.shape[0]}, {self.data_y.shape[0]}, and {self.data_z.shape[0]}."
            )
        # Normalize the data
        self.normalize = normalize
        if self.normalize and (
            self.data_x.ndim != 1 or self.data_y.ndim != 1 or self.data_z.ndim != 1
        ):
            raise ValueError("Data arrays must be 1D for normalization.")
        if self.normalize:
            self.data_x = normalize_data_0_1(self.data_x)
            self.data_y = normalize_data_0_1(self.data_y)
            self.data_z = normalize_data_0_1(self.data_z)
        super().__init__(base=base)

    def _generic_cmi_from_entropy(
        self,
        estimator: type(EntropyEstimator) | type(MutualInformationEstimator),
        noise_level: float = 0,
        kwargs: dict = None,
    ) -> float:
        """Calculate the conditional mutual information with the entropy estimator.

        Conditional Mutual Information (CMI) between two random variables :math:`X` and
        :math:`Y` given a third variable :math:`Z` quantifies the amount of information
        obtained about one variable through the other, conditioned on the third.
        In terms of entropy (H), CMI is expressed as:

        .. math::

                I(X, Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)

        where :math:`H(X, Z)` is the joint entropy of :math:`X` and :math:`Z`,
        :math:`H(Y, Z)` is the joint entropy of :math:`Y` and :math:`Z`,
        :math:`H(X, Y, Z)` is the joint entropy of :math:`X`, :math:`Y`, and :math:`Z`,
        and :math:`H(Z)` is the entropy of :math:`Z`.

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
            The conditional mutual information between the two variables given the third.

        Notes
        -----
        If possible, estimators should use a dedicated conditional mutual information method.
        This helper method is provided as a generic fallback.
        """

        # Ensure source and dest are numpy arrays
        data_x = self.data_x.copy()
        data_y = self.data_y.copy()
        data_z = self.data_z.copy()

        # Add Gaussian noise to the data if the flag is set
        if noise_level:
            data_x = data_x if data_x.dtype == float else data_x.astype(float)
            data_y = data_y if data_y.dtype == float else data_y.astype(float)
            data_z = data_z if data_z.dtype == float else data_z.astype(float)
            data_x += self.rng.normal(0, noise_level, data_x.shape)
            data_y += self.rng.normal(0, noise_level, data_y.shape)
            data_z += self.rng.normal(0, noise_level, data_z.shape)

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
            h_x_z = estimator((self.data_x, self.data_z), **kwargs).global_val()
            h_y_z = estimator((self.data_y, self.data_z), **kwargs).global_val()
            h_x_y_z = estimator(
                (self.data_x, self.data_y, self.data_z), **kwargs
            ).global_val()
            h_z = estimator(self.data_z, **kwargs).global_val()
            return h_x_z + h_y_z - h_x_y_z - h_z
        else:
            raise ValueError(f"Estimator must be an EntropyEstimator, not {estimator}.")


class TransferEntropyEstimator(RandomGeneratorMixin, Estimator, ABC):
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
    .transfer_entropy.symbolic.SymbolicTEEstimator
    .transfer_entropy.tsallis.TsallisTEEstimator
    """

    def __init__(
        self,
        source,
        dest,
        prop_time: int = 0,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        step_size: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data."""
        if len(source) != len(dest):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(source)} and {len(dest)}."
            )
        if not isinstance(prop_time, int):
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
        # Permutation flag - used by the p-value method and te_observations slicing
        self.permute_src = False
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

                I(Y_{t_{n+1}} : X_{t_n}^{(l)} | Y_{t_n}^{(k)}) = H(Y_{t_{n+1}}, Y_{t_n}^{(k)}) + H(X_{t_n}^{(l)}, Y_{t_n}^{(k)}) - H(Y_{t_{n+1}}, X_{t_n}^{(l)}, Y_{t_n}^{(k)}) - H(Y_{t_n}^{(k)})

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
        )

        h_y_history_y_future = estimator(marginal_2_space_data, **kwargs).global_val()
        h_x_history_y_history = estimator(marginal_1_space_data, **kwargs).global_val()
        h_x_history_y_history_y_future = estimator(
            joint_space_data, **kwargs
        ).global_val()
        h_y_history = estimator(dest_past_embedded, **kwargs).global_val()

        # Compute Transfer Entropy
        return (
            h_y_history_y_future
            + h_x_history_y_history
            - h_x_history_y_history_y_future
            - h_y_history
        )


class PValueMixin(RandomGeneratorMixin):
    """Mixin for p-value calculation.

    There are two methods to calculate the p-value:

    - Permutation test: shuffle the data and calculate the measure.
    - Bootstrap: resample the data and calculate the measure.  # TODO: Implement

    The :func:`permutation_test` can be used to determine a p-value for a measure,
    the :func:`calculate_permuted` additionally for effective TE.

    To be used as a mixin class with other :class:`Estimator` Estimator classes.
    Inherit before the main class.

    Data attribute to be shuffled depends on the derived class.

    - For entropy estimators, it is `data`.
    - For mutual information estimators, it is `data_x`.
    - For transfer entropy estimators, it is `source`.

    Attributes
    ----------
    permutation_data_attribute : str
        The attribute to shuffle.

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
        self.p_val = None
        self.p_val_method = None
        super().__init__(*args, **kwargs)
        # Determine the attribute to shuffle
        if isinstance(self, EntropyEstimator):
            self.permutation_data_attribute = "data"
        elif isinstance(self, MutualInformationEstimator):
            self.permutation_data_attribute = "data_y"
        elif isinstance(self, TransferEntropyEstimator):
            self.permutation_data_attribute = "dest"
        else:
            raise NotImplementedError(
                "P-value method is not implemented for the estimator."
            )

    def p_value(self, *args, method=Config.get("p_value_method"), **kwargs) -> float:
        """Calculate the p-value of the measure.

        Method can be "permutation_test" or "bootstrap".

        Parameters
        ----------
        method : str
            The method to calculate the p-value.
            Default is the value set in the configuration.
        n_permutations : int
            For permutation test, the number of permutations to perform.
            Needs to be a positive integer.

        Returns
        -------
        p_value : float
            The p-value of the measure.

        Raises
        ------
        ValueError
            If the chosen method is unknown.
        """
        if self.p_val is not None and method == self.p_val_method:
            return self.p_val
        logger.debug(
            f"Calculating the p-value of the measure {self.__class__.__name__} "
            f"using the {method} method."
        )
        self.p_val_method = method
        if method == "permutation_test":  # Permutation test
            if self.permutation_data_attribute == "source":
                self.p_val = self.permutation_test_te(*args, **kwargs)
            else:  # entropy or mutual information
                self.p_val = self.permutation_test(*args, **kwargs)
        elif method == "bootstrap":
            raise NotImplementedError("Bootstrap method is not implemented.")  # TODO
        else:
            raise ValueError(f"Invalid p-value method: {method}.")

        return self.p_val

    def calculate_permuted(self):
        """Calculate the measure for the permuted data.

        Returns
        -------
        float
            The measure for the permuted data.
        """
        # Backup the original data
        original_data = getattr(self, self.permutation_data_attribute).copy()
        # Calculate the measure for the permuted data
        result = self._calculate_permuted()
        # Restore the original data
        setattr(self, self.permutation_data_attribute, original_data)
        return result

    def _calculate_permuted(self):
        """Calculate the measure for the permuted data."""
        # Shuffle the data
        setattr(
            self,
            self.permutation_data_attribute,
            self.rng.permutation(getattr(self, self.permutation_data_attribute)),
        )
        # Calculate the measure
        res_permuted = self._calculate()
        return (
            res_permuted if isinstance(res_permuted, float) else np_mean(res_permuted)
        )

    def permutation_test(self, n_permutations: int) -> float:
        """Calculate the permutation test.

        Parameters
        ----------
        n_permutations : int
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
        if not isinstance(n_permutations, int) or n_permutations < 1:
            raise ValueError(
                "Number of permutations must be a positive integer, "
                f"not {n_permutations} ({type(n_permutations)})."
            )
        # Store unshuffled data
        original_data = getattr(self, self.permutation_data_attribute).copy()
        # Perform permutations
        permuted_values = [self._calculate_permuted() for _ in range(n_permutations)]
        if isinstance(permuted_values[0], tuple):
            permuted_values = [x[0] for x in permuted_values]
        # Restore the original data
        setattr(self, self.permutation_data_attribute, original_data)
        # Calculate the p-value
        return np_sum(array(permuted_values) >= self.global_val()) / n_permutations

    def permutation_test_te(self, n_permutations: int) -> float:
        """Calculate the permutation test for transfer entropy.

        Parameters
        ----------
        n_permutations : int
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
        if not isinstance(n_permutations, int) or n_permutations < 1:
            raise ValueError(
                "Number of permutations must be a positive integer, "
                f"not {n_permutations} ({type(n_permutations)})."
            )
        # Activate the permutation flag to permute the source data when slicing
        self.permute_src = self.rng
        permuted_values = [self._calculate() for _ in range(n_permutations)]
        if isinstance(permuted_values[0], tuple):
            permuted_values = [x[0] for x in permuted_values]
        # Deactivate the permutation flag
        self.permute_src = False
        return np_sum(array(permuted_values) >= self.global_val()) / n_permutations


class EffectiveValueMixin(PValueMixin):
    """Mixin for effective value calculation.

    To be used as a mixin class with :class:`TransferEntropyEstimator`
    and :class:`MutualInformationEstimator` derived classes.
    Inherit before the main class.

    Attributes
    ----------
    res_effective : float | None
        The effective transfer entropy/mutual information.

    Notes
    -----
    The effective value is the difference between the original
    value and the value calculated for the permuted data.
    This adds the :class:`PValueMixin` for the permutation test.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the estimator with the effective value."""
        self.res_effective = None
        super().__init__(*args, **kwargs)

    def effective_val(self):
        """Return the effective value (MI/TE).

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
        global_permuted = self.calculate_permuted()
        if isinstance(global_permuted, tuple):
            return self.global_val() - global_permuted[0]
        return self.global_val() - global_permuted
