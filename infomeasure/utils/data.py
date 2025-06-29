"""Data structures and containers for the infomeasure package."""

from dataclasses import dataclass, field

from numpy import ndarray, unique, asarray, percentile, integer, issubdtype


@dataclass(frozen=True)
class DiscreteData:
    """Container for discrete random variable data.

    Attributes
    ----------
    uniq : ndarray
        Array of unique values in the discrete data.
    counts : ndarray
        Array of counts for each unique value.
    data : ndarray, optional
        Original data array. Defaults to None.
    N : int, optional
        Total number of samples (length of data). Defaults to None.
    K : int, optional
        Number of unique values (len(uniq)). Defaults to None.
    """

    uniq: ndarray
    counts: ndarray
    N: int = field(init=False)
    K: int = field(init=False)
    data: ndarray = None

    @classmethod
    def from_data(cls, data: ndarray) -> "DiscreteData":
        """Create a DiscreteData object from a data array.

        Parameters
        ----------
        data : ndarray
            Raw data array to analyse.

        Returns
        -------
        DiscreteData
            New instance with computed unique values and counts.
        """
        data = asarray(data)
        if len(data) == 0:
            raise ValueError(
                f"`data` must not be empty, got {data} with shape {data.shape} instead."
            )
        uniq, counts = unique(data, return_counts=True)
        return cls(uniq=uniq, counts=counts, data=data)

    @classmethod
    def from_counts(cls, uniq: ndarray, counts: ndarray) -> "DiscreteData":
        """
        Constructs a `DiscreteData` instance from unique values and their counts.

        This class method creates an instance of the `DiscreteData` class by
        specifying the unique values and their corresponding counts. The `data`
        attribute of the created instance is set to `None`.

        Parameters
        ----------
        uniq : ndarray
            An array of unique values.
        counts : ndarray
            An array of counts corresponding to the unique values.

        Returns
        -------
        DiscreteData
            A new instance of the `DiscreteData` class initialized with the
            given unique values and their counts.
        """
        return cls(uniq=asarray(uniq), counts=asarray(counts))

    def __post_init__(self):
        """Validate attributes and set N and K."""
        if len(self.uniq) != len(self.counts):
            raise ValueError("uniq and counts must have same length")
        if any(self.counts < 0):
            raise ValueError("counts must be non-negative")
        if self.counts.sum() == 0:
            raise ValueError("counts must sum to a positive value")
        if not issubdtype(self.counts.dtype, integer):
            raise ValueError("counts must be integers")

        # Set K (number of unique values)
        object.__setattr__(self, "K", len(self.uniq))

        # Set N if data available
        if self.data is not None:
            if len(self.data) == 0:
                raise ValueError("data must not be empty")
            # Consistency between data and counts
            if self.counts.sum() != len(self.data):
                raise ValueError("counts must sum to the length of data")
            object.__setattr__(self, "N", len(self.data))
        else:
            object.__setattr__(self, "N", int(self.counts.sum()))

    @property
    def probabilities(self) -> ndarray:
        """
        Computes and returns the probabilities by normalizing counts.

        The `probabilities` property calculates the probabilities as the ratio
        of counts to the total value `N`. This provides a normalized
        representation of counts as probabilities.

        Returns
        -------
        ndarray
            An array containing the probabilities, calculated by dividing
            counts by `N`.
        """
        return self.counts / self.N

    @property
    def distribution_dict(self) -> dict:
        """Dictionary mapping unique elements to their corresponding probabilities.

        Returns
        -------
        dict
            A dictionary where keys are unique elements and values are their
            corresponding probabilities.
        """
        return dict(zip(self.uniq, self.probabilities))


@dataclass(frozen=True)
class StatisticalTestResult:
    """Comprehensive statistical test result containing *p*-value, *t*-score,
    and confidence intervals.

    Attributes
    ----------
    p_value : float
        The *p*-value of the statistical test.
    t_score : float
        The *t*-score (standardized test statistic).
    test_values : ndarray
        The test values from permutation/bootstrap sampling.
    observed_value : float
        The observed value being tested.
    null_mean : float
        Mean of the null distribution (test values).
    null_std : float
        Standard deviation of the null distribution.
    n_tests : int
        Number of tests performed (permutations or bootstrap samples).
    method : str
        The statistical test method used ("permutation_test" or "bootstrap").
    """

    p_value: float
    t_score: float
    test_values: ndarray
    observed_value: float
    null_mean: float
    null_std: float
    n_tests: int
    method: str

    def percentile(self, q, method="linear"):
        """Compute the q-th percentile of the test values.

        This method wraps numpy's percentile function to compute percentiles
        of the test values from the statistical test.

        Parameters
        ----------
        q : array_like of float
            Percentage or sequence of percentages for the percentiles to compute.
            Values must be between 0 and 100 inclusive.
        method : str, optional
            Method to use for estimating the percentile. Default is "linear".
            See :py:func:`numpy.percentile` for available methods.

        Returns
        -------
        percentile : scalar or ndarray
            If `q` is a single percentile, returns a scalar.
            If multiple percentiles are given, returns an array.

        See Also
        --------
        numpy.percentile : Compute percentiles along specified axes.

        Notes
        -----
        For details on the method parameter, reference :py:func:`numpy.percentile`.

        Examples
        --------
        >>> result = estimator.statistical_test(n_tests=100,method="permutation_test")
        >>> result.percentile(50)  # Median
        >>> result.percentile([25, 75])  # Quartiles
        >>> result.percentile(95, method="nearest")  # 95th percentile with nearest method
        """
        return percentile(self.test_values, q, method=method)

    def confidence_interval(self, confidence_level, method="linear"):
        """Get confidence interval for the specified confidence level.

        This is a convenience function that converts a confidence level
        (e.g., 95 for 95% CI) to the appropriate percentile calls.

        Parameters
        ----------
        confidence_level : float
            Confidence level as a percentage (e.g., 95 for 95% CI).
            Must be between 0 and 100.
        method : str, optional
            Method to use for estimating the percentile. Default is "linear".
            See :py:func:`numpy.percentile` for available methods.

        Returns
        -------
        ndarray
            Array containing [lower_bound, upper_bound] of the confidence interval.

        Raises
        ------
        ValueError
            If confidence_level is not between 0 and 100.

        Examples
        --------
        >>> result = estimator.statistical_test(n_tests=100)
        >>> result.confidence_interval(95)  # 95% CI
        >>> result.confidence_interval(90, method="nearest")  # 90% CI with nearest method
        """
        # Validate confidence level
        if not 0 < confidence_level < 100:
            raise ValueError(
                f"Confidence level must be between 0 and 100, got {confidence_level}"
            )

        # Calculate percentiles: y = (100 - x) / 2
        y = (100 - confidence_level) / 2
        percentiles = [y, 100 - y]

        return self.percentile(percentiles, method=method)
