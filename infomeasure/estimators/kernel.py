"""Kernel-based estimators for information measures."""

import numpy as np
from scipy.stats import gaussian_kde


def kde_probability_density_function(data, x, bandwidth, kernel="box"):
    """
    Estimate the probability density function at point x using Kernel Density Estimation
    with a given kernel.


    Parameters
    ----------
    data : array
        A numpy array of data points, where each column represents a dimension.
    x : array-like
        Point at which to estimate the probability density.
    bandwidth : float
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use ('gaussian' or 'box').

    Returns
    -------
    float
        Estimated probability density at point x.
    """
    if kernel == "gaussian":
        kde = gaussian_kde(data.T, bw_method=bandwidth)
        return kde.evaluate(x)
    elif kernel == "box":
        # Convert x to an array to ensure compatibility with numpy operations
        x = np.asarray(x)
        # Make sure x is 2D for consistent numpy subtraction operation
        # if x.ndim == 1:
        # x = x.reshape((-1, 1))
        # Define the box kernel density estimation
        N, d = data.shape
        # Compute the scaled data by the bandwidth and check if it falls within the unit hypercube centered at x
        scaled_data = np.abs(data - x) / bandwidth
        within_box = np.all(scaled_data <= 0.5, axis=1)
        # Count the number of points inside the box
        count = np.sum(within_box)
        # Normalize by the number of points and the volume of the box (bandwidth^dimension)
        volume = bandwidth**d
        return count / (N * volume)
    else:  # TODO: Add more kernel types as needed
        raise ValueError(f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'.")


def entropy(data, bandwidth: float | int, kernel: str = "box"):
    """
    Compute the entropy of the distribution given by kernel density estimation.

    Parameters
    ----------
    data : array
        A numpy array of data points, where each column represents a dimension.
    bandwidth : float
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, see :func:`kde_probability_density_function`.

    Returns
    -------
    entropy : float
        The entropy of the estimated distribution.

    Notes
    -----
    The entropy is computed as the negative mean of the log densities.
    Ensure that the data is passed as a 2D array with shape (N, d) for d dimensions.
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    densities = np.array(
        [
            kde_probability_density_function(data, data[i], bandwidth, kernel=kernel)
            for i in range(data.shape[0])
        ]
    )

    # Replace densities of 0 with a small number to avoid log(0)
    # TODO: Make optional
    densities[densities == 0] = np.finfo(float).eps

    # Compute the log of the densities
    log_densities = np.log(densities)

    # Compute the entropy
    entropy = -np.mean(log_densities)

    return entropy


def normalize_data_0_1(data):
    """
    Normalize the given data to scale each dimension to the range [0, 1].

    Parameters
    ----------
    data : 2D array-like
        List of data points where each row is a data point in d-dimensional space.

    Returns
    -------
    2D array
        Data scaled to the range [0, 1].

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> normalize_data_0_1(data)
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)


def mutual_information(
    source_data, dest_data, bandwidth, time_delay=0, normalize=False, kernel="box"
):
    """
    Compute the mutual information between two datasets using kernel density estimation.

    Additional parameters allow for time delay and normalization of the data.

    Parameters
    ----------
    source_data : array
        A numpy array of data points for the source variable.
    dest_data : array
        A numpy array of data points for the destination variable.
    bandwidth : float
        The bandwidth for the kernel.
    time_delay : int
        Time delay to apply to the destination data relative to the source data.
    normalize : bool
        If True, normalize the data before analysis.
    kernel : str
        Type of kernel to use, see :func:`kde_probability_density_function`.

    Returns
    -------
    local_mi_values : array
        The mutual information between the two datasets.
    average_mi : float
        The average mutual information between the two datasets.
    std_mi : float
        The standard deviation of the mutual information between the two datasets.

    """
    # Ensure source_data and dest_data are 2D arrays
    if source_data.ndim == 1:
        source_data = source_data[:, np.newaxis]
    if dest_data.ndim == 1:
        dest_data = dest_data[:, np.newaxis]

    # Normalize if necessary
    if normalize:
        source_data = normalize_data_0_1(source_data)
        dest_data = normalize_data_0_1(dest_data)

    # Apply time delay
    if time_delay > 0:
        source_data = source_data[:-time_delay]
        dest_data = dest_data[time_delay:]
    elif time_delay < 0:
        source_data = source_data[-time_delay:]
        dest_data = dest_data[:time_delay]

    # Combine source and dest data for joint density estimation
    # joint_data = np.vstack([source_data.ravel(), dest_data.ravel()]).T

    # Combine source and dest data for joint density estimation
    joint_data = np.hstack([source_data, dest_data])

    # Compute joint density using KDE for each point in the joint data
    joint_density = np.array(
        [
            kde_probability_density_function(
                joint_data, joint_data[i], bandwidth, kernel
            )
            for i in range(joint_data.shape[0])
        ]
    )

    # Compute individual densities for source and dest data
    source_density = np.array(
        [
            kde_probability_density_function(
                source_data, source_data[i], bandwidth, kernel
            )
            for i in range(source_data.shape[0])
        ]
    )
    dest_density = np.array(
        [
            kde_probability_density_function(dest_data, dest_data[i], bandwidth, kernel)
            for i in range(dest_data.shape[0])
        ]
    )

    # Avoid division by zero or log of zero by replacing zeros with a small positive value
    # TODO: Make optional
    joint_density[joint_density == 0] = np.finfo(float).eps
    source_density[source_density == 0] = np.finfo(float).eps
    dest_density[dest_density == 0] = np.finfo(float).eps

    # Compute mutual information
    # mi = np.mean(np.log(joint_density / (source_density * dest_density)))
    # return mi

    # New section for computing and returning local MI, mean, and Std
    local_mi_values = np.log(joint_density / (source_density * dest_density))
    average_mi = np.mean(local_mi_values)  # Global mutual information
    std_mi = np.std(local_mi_values)  # Standard deviation of local mutual information

    return local_mi_values, average_mi, std_mi


def _data_slice(
    data_source,
    data_destination,
    src_hist_len=1,
    dest_hist_len=1,
):
    """
    Slice the data arrays to prepare for kernel density estimation.

    Parameters
    ----------
    data_source : array
        A numpy array of data points for the source variable.
    data_destination : array
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
    N = len(data_source)
    # Prepare multivariate data arrays for KDE: Numerators
    numerator_term1 = np.column_stack(
        (
            data_destination[max(dest_hist_len, src_hist_len) : N],
            np.array(
                [
                    data_destination[i - dest_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
            np.array(
                [
                    data_source[i - src_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
        )
    )

    numerator_term2 = np.array(
        [
            data_destination[i - dest_hist_len : i]
            for i in range(max(dest_hist_len, src_hist_len), N)
        ]
    )

    # Prepare for KDE: Denominators
    denominator_term1 = np.column_stack(
        (
            np.array(
                [
                    data_destination[i - dest_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
            np.array(
                [
                    data_source[i - src_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
        )
    )

    denominator_term2 = np.column_stack(
        (
            data_destination[max(dest_hist_len, src_hist_len) : N],
            np.array(
                [
                    data_destination[i - dest_hist_len : i]
                    for i in range(max(dest_hist_len, src_hist_len), N)
                ]
            ),
        )
    )

    return numerator_term1, numerator_term2, denominator_term1, denominator_term2


def transfer_entropy(
    data_source,
    data_destination,
    src_hist_len=1,
    dest_hist_len=1,
    bandwidth=0.3,
    kernel="box",
):
    """
    Compute the transfer entropy from source to destination data using
    kernel density estimation.

    Parameters
    ----------
    data_source : array
        A numpy array of data points for the source variable.
    data_destination : array
        A numpy array of data points for the destination variable.
    src_hist_len : int
        Number of past observations to consider for the source data.
    dest_hist_len : int
        Number of past observations to consider for the destination data.
    bandwidth : float
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, see :func:`kde_probability_density_function`.

    Returns
    -------
    local_te_values : array
        Local transfer entropy values.
    average_te : float
        The average transfer entropy value.
    """
    N = len(data_source)
    local_te_values = np.zeros(N - max(src_hist_len, dest_hist_len))

    # Prepare multivariate data arrays for KDE: Numerators
    numerator_term1, numerator_term2, denominator_term1, denominator_term2 = (
        _data_slice(data_source, data_destination, src_hist_len, dest_hist_len)
    )

    # Compute KDE for each term directly using slices
    for i in range(len(local_te_values)):
        # g(x_{i+1}, x_i^{(k)}, y_i^{(l)})
        p_x_future_x_past_y_past = kde_probability_density_function(
            numerator_term1, numerator_term1[i], bandwidth, kernel
        )
        # g(x_i^{(k)})
        p_x_past = kde_probability_density_function(
            numerator_term2, numerator_term2[i], bandwidth, kernel
        )
        # g(x_i^{(k)}, y_i^{(l)})
        p_xy_past = kde_probability_density_function(
            denominator_term1, denominator_term1[i], bandwidth, kernel
        )
        # g(x_{i+1}, x_i^{(k)})
        p_x_future_x_past = kde_probability_density_function(
            denominator_term2, denominator_term2[i], bandwidth, kernel
        )

        # Calculate local TE value
        if (
            p_x_future_x_past_y_past * p_x_past > 0
            and p_xy_past * p_x_future_x_past > 0
        ):
            local_te_values[i] = np.log(
                p_x_future_x_past_y_past * p_x_past / (p_xy_past * p_x_future_x_past)
            )

    # Calculate average TE
    average_te = np.nanmean(local_te_values)  # Using nanmean to ignore any NaNs

    return local_te_values, average_te
