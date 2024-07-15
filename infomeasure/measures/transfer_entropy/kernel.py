"""Module for the kernel-based transfer entropy estimator."""

from numpy import array, column_stack, nanmean, zeros

from ... import Config
from ...utils.types import LogBaseType
from ..base import LogBaseMixin, TransferEntropyEstimator
from ..utils.kde import kde_probability_density_function


class KernelTEEstimator(LogBaseMixin, TransferEntropyEstimator):
    """Estimator for transfer entropy using Kernel Density Estimation (KDE).

    Attributes
    ----------
    source, dest : array-like
        The source and destination data used to estimate the transfer entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Methods
    -------
    calculate()
        Calculate the transfer entropy from source to destination.
    """

    def __init__(
        self,
        source,
        dest,
        bandwidth: float | int,
        kernel: str,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data and parameters.

        Parameters
        ----------
        bandwidth : float | int
            The bandwidth for the kernel.
        kernel : str
            Type of kernel to use, compatible with the KDE
            implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
        src_hist_len, dest_hist_len : int
            Number of past observations to consider for the source and destination data.
        """
        super().__init__(source, dest, base=base)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.src_hist_len = src_hist_len
        self.dest_hist_len = dest_hist_len

    def calculate(self):
        """Calculate the transfer entropy of the data.

        Returns
        -------
        local_te_values : array
            Local transfer entropy values.
        average_te : float
            The average transfer entropy value.
        """
        N = len(self.source)
        local_te_values = zeros(N - max(self.src_hist_len, self.dest_hist_len))

        # Prepare multivariate data arrays for KDE: Numerators
        numerator_term1, numerator_term2, denominator_term1, denominator_term2 = (
            _data_slice(self.source, self.dest, self.src_hist_len, self.dest_hist_len)
        )

        # Compute KDE for each term directly using slices
        for i in range(len(local_te_values)):
            # g(x_{i+1}, x_i^{(k)}, y_i^{(l)})
            p_x_future_x_past_y_past = kde_probability_density_function(
                numerator_term1, numerator_term1[i], self.bandwidth, self.kernel
            )
            # g(x_i^{(k)})
            p_x_past = kde_probability_density_function(
                numerator_term2, numerator_term2[i], self.bandwidth, self.kernel
            )
            # g(x_i^{(k)}, y_i^{(l)})
            p_xy_past = kde_probability_density_function(
                denominator_term1, denominator_term1[i], self.bandwidth, self.kernel
            )
            # g(x_{i+1}, x_i^{(k)})
            p_x_future_x_past = kde_probability_density_function(
                denominator_term2, denominator_term2[i], self.bandwidth, self.kernel
            )

            # Calculate local TE value
            if (
                p_x_future_x_past_y_past * p_x_past > 0
                and p_xy_past * p_x_future_x_past > 0
            ):
                local_te_values[i] = self._log_base(
                    p_x_future_x_past_y_past
                    * p_x_past
                    / (p_xy_past * p_x_future_x_past)
                )

        # Calculate average TE
        average_te = nanmean(local_te_values)  # Using nanmean to ignore any NaNs

        return local_te_values, average_te


def _data_slice(
    source,
    destination,
    src_hist_len=1,
    dest_hist_len=1,
):
    """
    Slice the data arrays to prepare for kernel density estimation.

    Parameters
    ----------
    source : array
        A numpy array of data points for the source variable.
    destination : array
        A numpy array of data points for the destination variable.
    src_hist_len : int
        Number of past observations to consider for the source data.
    dest_hist_len : int
        Number of past observations to consider for the destination data.

    Returns
    -------
    numerator_term1 : array
    numerator_term2 : array
    denominator_term1 : array
    denominator_term2 : array
    """
    N = len(source)
    # Prepare multivariate data arrays for KDE: Numerators
    numerator_term1 = column_stack(
        (
            destination[max(dest_hist_len, src_hist_len) : N],
            array(
                [
                    destination[i - dest_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
            array(
                [
                    source[i - src_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
        )
    )

    numerator_term2 = array(
        [
            destination[i - dest_hist_len : i]
            for i in range(max(dest_hist_len, src_hist_len), N)
        ]
    )

    # Prepare for KDE: Denominators
    denominator_term1 = column_stack(
        (
            array(
                [
                    destination[i - dest_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
            array(
                [
                    source[i - src_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
        )
    )

    denominator_term2 = column_stack(
        (
            destination[max(dest_hist_len, src_hist_len) : N],
            array(
                [
                    destination[i - dest_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
        )
    )

    return numerator_term1, numerator_term2, denominator_term1, denominator_term2
