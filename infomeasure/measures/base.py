"""Module containing the base classes for the measure estimators."""

from abc import ABC, abstractmethod

from numpy import array, log, log2, log10
from numpy import sum as np_sum
from numpy.random import default_rng

from .. import Config
from ..utils.types import LogBaseType


class Estimator(ABC):
    """Abstract base class for all measure estimators.

    Methods
    -------
    calculate()
        Calculate the measure.
    p_value()
        Calculate the p-value of the measure.

    See Also
    --------
    EntropyEstimator, MutualInformationEstimator, TransferEntropyEstimator

    Notes
    -----
    The `calculate` method needs to be implemented in the derived classes.
    If the measure has a p-value, the `p_value` method should be implemented.
    """

    @abstractmethod
    def calculate(self):
        """Calculate the measure."""
        pass

    def p_value(self, *args, **kwargs) -> float:
        """Calculate the p-value of the measure. Not implemented."""
        pass


class EntropyEstimator(Estimator, ABC):
    """Abstract base class for entropy estimators.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.

    Methods
    -------
    calculate()
        Calculate the entropy.

    See Also
    --------
    .entropy.discrete.DiscreteEntropyEstimator
    """

    def __init__(self, data):
        """Initialize the estimator with the data."""
        self.data = data


class MutualInformationEstimator(Estimator, ABC):
    """Abstract base class for mutual information estimators.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information. The data should be
        of the same length.

    Methods
    -------
    calculate()
        Calculate the mutual information.

    Raises
    ------
    ValueError
        If the data arrays are not of the same length.

    See Also
    --------
    .mutual_information.discrete.DiscreteMIEstimator
    """

    def __init__(self, data_x, data_y):
        """Initialize the estimator with the data."""
        if len(data_x) != len(data_y):
            raise ValueError(
                "Data arrays must be of the same length, "
                f"not {len(data_x)} and {len(data_y)}."
            )
        self.data_x = data_x
        self.data_y = data_y


class TransferEntropyEstimator(Estimator, ABC):
    """Abstract base class for transfer entropy estimators.

    Attributes
    ----------
    source : array-like
        The source data used to estimate the transfer entropy.
    dest : array-like
        The destination data used to estimate the transfer entropy.

    Methods
    -------
    calculate()
        Calculate the transfer entropy.

    See Also
    --------
    .transfer_entropy.discrete.DiscreteTEEstimator
    """

    def __init__(self, source, dest):
        """Initialize the estimator with the data."""
        self.source = source
        self.dest = dest


class LogBaseMixin:
    """Mixin for logarithmic base calculation.

    To be used as a mixin class with other :class:`Estimator` Estimator classes.
    Inherit before the main class.

    Attributes
    ----------
    base : int
        The logarithm base for the measure calculation.
    """

    def __init__(self, *args, base: LogBaseType = Config.get("base"), **kwargs):
        """Initialize the estimator with the base."""
        self.base = base
        super().__init__(*args, **kwargs)

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


class PermutationTestMixin(RandomGeneratorMixin):
    """Mixin for permutation test calculation.

    The :func:`permutation_test` can be used to determine a p-value for a measure,
    the :func:`calculate_permuted` for an effect size.

    To be used as a mixin class with other :class:`Estimator` Estimator classes.
    Inherit before the main class.

    Data attribute to be shuffled depends on the derived class.
    - For entropy estimators, it is `data`.
    - For mutual information estimators, it is `data_x`.
    - For transfer entropy estimators, it is `source`.

    Attributes
    ----------
    key : str
        The attribute to shuffle.

    Methods
    -------
    calculate_permuted()
        Calculate the measure for the permuted data.
    permutation_test()
        Calculate the permutation test.

    Notes
    -----
    The permutation test is a non-parametric statistical test to determine if the
    observed effect is significant. The null hypothesis is that the measure is
    not different from random, and the p-value is the proportion of permuted
    measures greater than the observed measure.

    Raises
    ------
    NotImplementedError
        If the permutation test is not implemented for the derived class.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the permutation test."""
        super().__init__(*args, **kwargs)
        # Determine the attribute to shuffle
        if isinstance(self, EntropyEstimator):
            self.key = "data"
        elif isinstance(self, MutualInformationEstimator):
            self.key = "data_x"
        elif isinstance(self, TransferEntropyEstimator):
            self.key = "source"
        else:
            raise NotImplementedError(
                "Permutation test not implemented for this estimator."
            )

    def calculate_permuted(self):
        """Calculate the measure for the permuted data.

        Returns
        -------
        float
            The measure for the permuted data.
        """
        # Backup the original data
        original_data = getattr(self, self.key).copy()
        # Calculate the measure for the permuted data
        result = self._calculate_permuted()
        # Restore the original data
        setattr(self, self.key, original_data)
        return result

    def _calculate_permuted(self):
        """Calculate the measure for the permuted data."""
        # Shuffle the data
        setattr(self, self.key, self.rng.permutation(getattr(self, self.key)))
        # Calculate the measure
        yield self.calculate()

    def permutation_test(self, num_permutations: int) -> float:
        """Calculate the permutation test.

        Parameters
        ----------
        num_permutations : int
            The number of permutations to perform.

        Returns
        -------
        float
            The p-value of the measure.
        """
        _, observed_global = self.calculate()
        # Store unshuffled data
        original_data = getattr(self, self.key).copy()
        # Perform permutations
        permuted_values = [
            self._calculate_permuted()[1] for _ in range(num_permutations)
        ]
        # Restore the original data
        setattr(self, self.key, original_data)
        # Calculate the p-value
        return np_sum(array(permuted_values) >= observed_global) / num_permutations
