"""Mixin classes for estimators from .base.py."""

from io import UnsupportedOperation
from typing import Union, Callable

from numpy import (
    issubdtype,
    integer,
    ndarray,
    asarray,
    mean as np_mean,
    std,
    sum as np_sum,
    nan,
)
from numpy.random import default_rng

from infomeasure import Config
from infomeasure.utils.config import logger
from infomeasure.utils.data import StatisticalTestResult


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


class StatisticalTestingMixin(RandomGeneratorMixin):
    """Mixin for comprehensive statistical testing including *p*-values, *t*-scores,
    and confidence intervals.

    There are two methods to perform statistical tests:

    - Permutation test: shuffle the data and calculate the measure.
    - Bootstrap: resample the data and calculate the measure.

    The :func:`statistical_test` method provides comprehensive statistical analysis
    including *p*-value, *t*-score, and confidence intervals in a single call.

    To be used as a mixin class with other :class:`Estimator` Estimator classes.
    Inherit before the main class.

    Notes
    -----
    The permutation test is a non-parametric statistical test to determine if the
    observed effect is significant. The null hypothesis is that the measure is
    not different from random, and the *p*-value is the proportion of permuted
    measures greater than the observed measure.

    Confidence intervals are calculated using percentiles of the null distribution
    from the resampling procedure.

    Raises
    ------
    NotImplementedError
        If the statistical test is not implemented for the estimator.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the statistical test mixin."""
        self.original_data = None
        super().__init__(*args, **kwargs)
        if not any(
            name in [cls.__name__ for cls in self.__class__.__mro__]
            for name in [
                "MutualInformationEstimator",
                "ConditionalMutualInformationEstimator",
                "TransferEntropyEstimator",
                "ConditionalTransferEntropyEstimator",
            ]
        ):
            raise NotImplementedError(
                "Statistical test is not implemented for the estimator."
            )

    def statistical_test(
        self, n_tests: int = None, method: str = None
    ) -> StatisticalTestResult:
        """Perform comprehensive statistical test including *p*-value, *t*-score,
        and confidence intervals.

        Method can be "permutation_test" or "bootstrap".

        - Permutation test: shuffle the data and calculate the measure.
        - Bootstrap: resample the data and calculate the measure.

        Parameters
        ----------
        n_tests : int, optional
            Number of permutations or bootstrap samples.
            Needs to be a positive integer.
            Default is the value set in the configuration.
        method : str, optional
            The method to calculate the statistical test.
            Options are "permutation_test" or "bootstrap".
            Default is the value set in the configuration.

        Returns
        -------
        StatisticalTestResult
            Comprehensive statistical test result containing *p*-value, *t*-score,
            and metadata. Percentiles can be calculated on demand using
            the percentile() method.

        Raises
        ------
        ValueError
            If the chosen method is unknown.
        UnsupportedOperation
            If the statistical test is not supported for the estimator type.
        """
        method, n_tests, test_values = self._statistical_test(method, n_tests)
        # Make a test result
        return self._statistical_test_result(
            observed_value=self.global_val(),
            test_values=test_values,
            n_tests=n_tests,
            method=method,
        )

    def _statistical_test(self, method, n_tests):
        # Set defaults
        if n_tests is None:
            n_tests = Config.get("statistical_test_n_tests")
        if method is None:
            method = Config.get("statistical_test_method")
        logger.debug(
            "Calculating statistical test "
            f"of the measure {self.__class__.__name__} "
            f"using the {method} method with {n_tests} tests."
        )
        # Validate inputs
        if not issubdtype(type(n_tests), integer) or n_tests < 1:
            raise ValueError(
                "Number of tests must be a positive integer, "
                f"not {n_tests} ({type(n_tests)})."
            )
        class_names = [cls.__name__ for cls in self.__class__.__mro__]
        if any(
            name in class_names
            for name in [
                "MutualInformationEstimator",
                "ConditionalMutualInformationEstimator",
            ]
        ):
            if len(self.data) != 2:
                raise UnsupportedOperation(
                    "Statistical test on mutual information is only supported "
                    "for two variables."
                )
            test_method = self._test_mi
        elif any(
            name in class_names
            for name in [
                "TransferEntropyEstimator",
                "ConditionalTransferEntropyEstimator",
            ]
        ):
            test_method = self._test_te
        else:
            raise NotImplementedError(
                "Statistical test is not implemented for this estimator."
            )
        # Generate test values and calculate comprehensive result
        test_values = test_method(n_tests, method)
        return method, n_tests, test_values

    @staticmethod
    def _statistical_test_result(
        observed_value: float,
        test_values: Union[ndarray, list, tuple],
        n_tests: int,
        method: str,
    ) -> StatisticalTestResult:
        """
        Calculate comprehensive statistical test result including *p*-value, *t*-score,
        and confidence intervals.

        Parameters
        ----------
        observed_value : float
            The observed value.
        test_values : array-like
            The test values from permutation/bootstrap sampling.
        n_tests : int
            Number of tests performed (permutations or bootstrap samples).
        method : str
            The statistical test method used ("permutation_test" or "bootstrap").

        Returns
        -------
        StatisticalTestResult
            Comprehensive statistical test result object.

        Raises
        ------
        ValueError
            If the observed value is not numeric.
        ValueError
            If the test values are not array-like.
        """
        # Input validation
        if not isinstance(observed_value, (int, float)):
            raise ValueError("Observed value must be numeric.")
        if not isinstance(test_values, (ndarray, list, tuple)):
            raise ValueError("Test values must be array-like.")
        if len(test_values) < 2:
            raise ValueError("Not enough test values for statistical test.")

        test_values = asarray(test_values)

        # Calculate basic statistics
        null_mean = np_mean(test_values)
        null_std = std(test_values, ddof=1)  # Unbiased estimator (dividing by N-1)

        # Compute *p*-value: proportion of test values greater than the observed value
        p_value = np_sum(test_values > observed_value) / len(test_values)

        # Compute *t*-score
        t_score = (observed_value - null_mean) / null_std if null_std > 0 else nan

        return StatisticalTestResult(
            p_value=p_value,
            t_score=t_score,
            test_values=test_values.copy(),
            observed_value=float(observed_value),
            null_mean=null_mean,
            null_std=null_std,
            n_tests=n_tests,
            method=method,
        )

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

    def _test_mi(self, n_tests: int, method: str) -> ndarray:
        """Generate test values for mutual information using permutation test or
        bootstrap.

        Parameters
        ----------
        n_tests : int
            The number of permutations or bootstrap samples.
        method : str
            The method to use ("permutation_test" or "bootstrap").

        Returns
        -------
        ndarray
            Array of test values from resampling.

        Raises
        ------
        ValueError
            If the method is invalid.
        """
        # Store unshuffled data
        self.original_data = self.data

        # Set up resampling method
        if method == "permutation_test":
            method_resample_src = lambda data_src: self.rng.permutation(
                data_src, axis=0
            )
        elif method == "bootstrap":
            method_resample_src = lambda data_src: self.rng.choice(
                data_src, size=data_src.shape[0], replace=True, axis=0
            )
        else:
            raise ValueError(f"Invalid statistical test method: {method}.")

        # Generate test values
        permuted_values = [
            self._calculate_mi_with_data_selection(method_resample_src)
            for _ in range(n_tests)
        ]

        # Restore the original data
        self.data = self.original_data
        return asarray(permuted_values)

    def _test_te(self, n_tests: int, method: str) -> ndarray:
        """Generate test values for transfer entropy using permutation test or
        bootstrap.

        Parameters
        ----------
        n_tests : int
            The number of permutations or bootstrap samples.
        method : str
            The method to use ("permutation_test" or "bootstrap").

        Returns
        -------
        ndarray
            Array of test values from resampling.

        Raises
        ------
        ValueError
            If the method is invalid.
        """
        # Set up resampling method
        if method == "permutation_test":
            self.permute_src = self.rng
            self.resample_src = False
        elif method == "bootstrap":
            self.permute_src = False
            self.resample_src = self.rng
        else:
            raise ValueError(f"Invalid statistical test method: {method}.")

        # Generate test values
        permuted_values = [self._calculate() for _ in range(n_tests)]
        if isinstance(permuted_values[0], ndarray):
            permuted_values = [np_mean(x) for x in permuted_values]

        # Deactivate the permutation/resample flags
        self.permute_src, self.resample_src = False, False
        return asarray(permuted_values)


class DiscreteMIMixin:
    """Mixin for handling discrete mutual information computations.

    Provides utilities and checks necessary for estimating discrete mutual
    information and conditional mutual information.
    Ensures that input data is suitable for these calculations and provides warnings
    when pre-processing steps, such as symbolizing or discretizing, are required.

    Attributes
    ----------
    data : Any
        The primary data to be used in mutual information estimation.
        It should be symbolized or discretized if it contains floating-point types.

    cond : Any, optional
        The conditional data for conditional mutual information estimation.
        If provided, it should also be symbolized or discretized if it contains
        floating-point types.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_data_mi(self):
        """Check the input data for discrete mutual information calculations.

        Verifies the types of the data attribute and condition attribute (if present)
        to ensure they are suitable for mutual information estimation. Warns if the
        data contains floating-point types and suggests appropriate transformations.

        Notes
        -----
        This method checks if the data attribute contains elements with a floating-point
        data type. If such types are detected, it logs a warning suggesting the need for
        symbolization or discretization for mutual information calculations. Similarly,
        if the `cond` attribute is present and has a floating-point type, the method logs
        a warning suggesting preprocessing for conditional mutual information calculations.
        This step ensures the validity and reliability of the mutual information estimation.

        Attributes
        ----------
        data : Any
            The primary data used in the computation. It must be symbolized or
            discretized for mutual information estimation if floating-point types
            are present.

        cond : Any, optional
            The conditional data used for conditional mutual information estimation.
            If present, it must also be symbolized or discretized if it contains
            elements of floating-point types.
        """
        if any(var.dtype.kind == "f" for var in self.data):
            logger.warning(
                "The data looks like a float array ("
                f"{[var.dtype for var in self.data]}). "
                "Make sure it is properly symbolized or discretized "
                "for the mutual information estimation."
            )
        if hasattr(self, "cond") and self.cond.dtype.kind == "f":
            logger.warning(
                "The conditional data looks like a float array ("
                f"{self.cond.dtype}). "
                "Make sure it is properly symbolized or discretized "
                "for the conditional mutual information estimation."
            )


class DiscreteTEMixin:
    """
    Mixin class for discrete transfer entropy calculations.

    Provides functionality to validate input data types for transfer entropy
    estimation processes. Ensures that source, destination, and conditional
    datasets are properly symbolized or discretized to prevent invalid results
    from using continuous floating-point data.

    Attributes
    ----------
    source : array-like
        The source data array utilized in transfer entropy calculations.
    dest : array-like
        The destination data array utilized in transfer entropy calculations.
    cond : array-like, optional
        The conditional data array utilized in transfer entropy calculations
        when applicable.
    """

    def _check_data_te(self):
        """Check the input data for discrete transfer entropy calculations.

        Checks the data types of the source, destination, and conditional data
        attributes involved in the transfer entropy estimation process.
        Issues warnings if any of these datasets are floating-point,
        as they may need proper symbolization or discretization in order to ensure the
        validity of the calculations.

        Notes
        -----
        Transfer entropy estimation requires input data to be symbolized or discretized,
        as raw continuous floating-point arrays may lead to incorrect results.
        This method specifically warns users when it detects floating-point arrays for
        critical data inputs (source, destination, or conditional data).

        Attributes
        ----------
        source : array-like
            The source data array whose data type is validated in this method.
        dest : array-like
            The destination data array whose data type is validated in this method.
        cond : array-like, optional
            The conditional data array whose data type is validated in this method if
            present.

        Warnings
        --------
        Issues a warning when the data type of `source` or `dest` is floating-point.
        If the conditional data array (`cond`) exists and its data type is
        floating-point, a separate warning will be issued.
        """
        if self.source.dtype.kind == "f" or self.dest.dtype.kind == "f":
            logger.warning(
                "The data looks like a float array ("
                f"source: {self.source.dtype}, dest: {self.dest.dtype}). "
                "Make sure the data is properly symbolized or discretized "
                "for the transfer entropy estimation."
            )

        if hasattr(self, "cond") and self.cond.dtype.kind == "f":
            logger.warning(
                "The conditional data looks like a float array ("
                f"{self.cond.dtype}). "
                "Make sure the data is properly symbolized or discretized "
                "for the conditional transfer entropy estimation."
            )


class EffectiveValueMixin(StatisticalTestingMixin):
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

    def effective_val(self, method: str = None):
        """Return the effective value.

        Calculates the effective value if not already done,
        otherwise returns the stored value.

        Returns
        -------
        effective : float
            The effective value.
        """
        _, _, test_values = self._statistical_test(n_tests=1, method=method)
        return self.global_val() - test_values[0]


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
