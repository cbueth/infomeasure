"""Module containing the base classes for the measure estimators."""

from abc import ABC, abstractmethod
from io import UnsupportedOperation
from typing import final

from numpy import std as np_std, asarray
from numpy import array, log, log2, log10
from numpy import sum as np_sum
from numpy.random import default_rng

from .. import Config
from ..utils.config import logger
from ..utils.types import LogBaseType
from .utils.normalize import normalize_data_0_1


class Estimator(ABC):
    """Abstract base class for all measure estimators.

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
        if isinstance(results, tuple):
            self.res_global, self.res_local = results
            logger.debug(
                f"Global: {self.res_global:.4e}, "
                # show the first max 5 local values
                f"Local: {', '.join([f'{x:.2e}' for x in self.res_local[:5]])}"
                f"{', ...' if len(self.res_local) > 5 else ''}"
            )
        else:
            self.res_global = results
            logger.debug(f"Global: {self.res_global:.4e}")

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
        result : tuple[float, array] | float
            Tuple of the global and local values,
            or just the global value if local values are not available.
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

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.


    See Also
    --------
    .entropy.discrete.DiscreteEntropyEstimator
    .entropy.kernel.KernelEntropyEstimator
    .entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator
    """

    def __init__(self, data, base: LogBaseType = Config.get("base")):
        """Initialize the estimator with the data."""
        self.data = asarray(data)
        super().__init__(base=base)


class MutualInformationEstimator(Estimator, ABC):
    """Abstract base class for mutual information estimators.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information. The data should be
        1D and of the same length.
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
        If the data arrays are not 1D or of different lengths.
    ValueError
        If the offset is not an integer.

    See Also
    --------
    .mutual_information.discrete.DiscreteMIEstimator
    .mutual_information.kernel.KernelMIEstimator
    .mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator
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
        if len(data_x) != len(data_y):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(data_x)} and {len(data_y)}."
            )
        if not isinstance(offset, int):
            raise ValueError(f"Offset must be an integer, not {offset}.")
        self.data_x = asarray(data_x)
        self.data_y = asarray(data_y)
        if data_x.ndim != 1 or data_y.ndim != 1:
            raise ValueError("Data arrays must be 1D.")
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
        if self.normalize:
            self.data_x = normalize_data_0_1(self.data_x)
            self.data_y = normalize_data_0_1(self.data_y)
        super().__init__(base=base)


class TransferEntropyEstimator(Estimator, ABC):
    """Abstract base class for transfer entropy estimators.

    Attributes
    ----------
    source : array-like
        The source data used to estimate the transfer entropy.
    dest : array-like
        The destination data used to estimate the transfer entropy.
    tau : int
        Time delay for state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from source to destination.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the data arrays are not 1D or of different lengths.

    See Also
    --------
    .transfer_entropy.discrete.DiscreteTEEstimator
    .transfer_entropy.kernel.KernelTEEstimator
    .transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator
    """

    def __init__(
        self,
        source,
        dest,
        offset: int = 0,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        tau: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data."""
        if len(source) != len(dest):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(source)} and {len(dest)}."
            )
        if not isinstance(offset, int):
            raise ValueError(f"Offset must be an integer, not {offset}.")
        self.source = asarray(source)
        self.dest = asarray(dest)
        if source.ndim != 1 or dest.ndim != 1:
            raise ValueError("Data arrays must be 1D.")
        # Apply the offset
        self.offset = offset
        if self.offset > 0:
            self.source = self.source[: -self.offset or None]
            self.dest = self.dest[self.offset :]
        elif self.offset < 0:
            self.source = self.source[-self.offset :]
            self.dest = self.dest[: self.offset or None]
        # Slicing parameters
        self.src_hist_len, self.dest_hist_len = src_hist_len, dest_hist_len
        self.tau = tau
        super().__init__(base=base)


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
            self.permutation_data_attribute = "data_x"
        elif isinstance(self, TransferEntropyEstimator):
            self.permutation_data_attribute = "source"
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
        if method == "permutation_test":
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
        return self._calculate()

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


class EffectiveTEMixin(PValueMixin):
    """Mixin for effective transfer entropy calculation.

    To be used as a mixin class with :class:`TransferEntropyEstimator`
    derived classes. Inherit before the main class.

    Attributes
    ----------
    res_effective : float | None
        The effective transfer entropy.

    Methods
    -------
    effective_val()
        Return the effective transfer entropy.

    Notes
    -----
    The effective transfer entropy is the difference between the original
    transfer entropy and the transfer entropy calculated for the permuted data.
    This adds the :class:`PValueMixin` for the permutation test.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the transfer entropy estimator with the effective TE."""
        self.res_effective = None
        super().__init__(*args, **kwargs)

    def effective_val(self):
        """Return the effective transfer entropy.

        Calculate the effective transfer entropy if not already calculated.

        Returns
        -------
        effective : float
            The effective transfer entropy.
        """
        if self.res_effective is None:
            self.res_effective = self._calculate_effective()
        return self.res_effective

    def _calculate_effective(self):
        """Calculate the effective transfer entropy.

        Returns
        -------
        effective : float
            The effective transfer entropy.
        """
        global_permuted = self.calculate_permuted()
        if isinstance(global_permuted, tuple):
            return self.global_val() - global_permuted[0]
        return self.global_val() - global_permuted
