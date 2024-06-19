"""Metric-based estimators for information measures."""

import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy.special import digamma

from ..utils.normalize import normalize_data_0_1


def kl_entropy_estimator(data, k=4, noise_level=1e-10):
    """
    Estimate the entropy of a dataset using the k-nearest neighbor method with the
    Kullback-Leibler divergence.

    Parameters
    ----------
    data : ndarray
        The dataset for which to estimate the entropy.
    k : int
        The number of nearest neighbors to consider.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.

    Returns
    -------
    entropy : float
        The estimated entropy of the dataset using the Kullback-Leibler divergence.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Add small Gaussian noise to data to avoid issues with zero distances
    noise = np.random.normal(0, noise_level, data.shape)
    data_noisy = data + noise

    # Build a cKDTree for efficient nearest neighbor search with maximum norm
    tree = cKDTree(data_noisy)  # cKDTree uses 'euclidean' metric by default

    # Find the k-th nearest neighbors for each point
    distances, _ = tree.query(
        data_noisy, k + 1, p=np.inf
    )  # Using p=np.inf to use Chebyshev distance

    # Exclude the zero distance to itself, which is the first distance
    distances = distances[:, k]

    # Constants for the entropy formula
    N = data.shape[0]
    d = data.shape[1]
    c_d = 1  # Volume of the d-dimensional unit ball for maximum norm

    # Compute the entropy estimator considering that the distances are already doubled
    entropy = (
        -digamma(k) + digamma(N) + np.log(c_d) + (d / N) * np.sum(np.log(2 * distances))
    )

    return entropy


def mutual_info_ksg_alg1(data_X, data_Y, k=4, noise=1e-8, normalize=True, time_delay=0):
    r"""
    Estimate mutual information between two datasets using the KSG method,
    with options for normalization, noise addition, and time delay.

    Parameters
    ----------
    data_X : array-like
        Observations for variable X.
    data_Y : array-like
        Observations for variable Y.
    k : int
        Number of nearest neighbors to consider.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.
    normalize : bool
        Flag to normalize the data to the [0, 1] range.
    time_delay : int
        Time delay to introduce between data_X and data_Y.

    Returns
    -------
    local_mi : array
        Local mutual information for each point.
    mi : float
        Estimated mutual information between the two datasets.
    """
    # Convert input data to numpy arrays for consistency
    data_X = np.asarray(data_X)
    data_Y = np.asarray(data_Y)

    # Normalize if necessary
    if normalize:
        data_X = normalize_data_0_1(data_X)
        data_Y = normalize_data_0_1(data_Y)

    # Apply time delay
    if time_delay > 0:
        data_X = data_X[:-time_delay]
        data_Y = data_Y[time_delay:]
    elif time_delay < 0:
        data_X = data_X[-time_delay:]
        data_Y = data_Y[:time_delay]

    # Add Gaussian noise to the data if the flag is set
    if noise:
        data_X += np.random.normal(0, noise, data_X.shape)
        data_Y += np.random.normal(0, noise, data_Y.shape)

    # Ensure the data is 2D for KDTree
    if data_X.ndim == 1:
        data_X = data_X[:, np.newaxis]
    if data_Y.ndim == 1:
        data_Y = data_Y[:, np.newaxis]

    # Stack the X and Y data to form joint observations
    data_joint = np.column_stack((data_X, data_Y))

    # Create a cKDTree for joint data to find nearest neighbors using the maximum norm
    tree_joint = cKDTree(data_joint, leafsize=10)  # defult leafsize is 10

    # Find the k-th nearest neighbor distance for each point in joint space using the
    # maximum norm
    distances, _ = tree_joint.query(data_joint, k=k + 1, p=np.inf)
    kth_distances = distances[:, -1]

    # Create cKDTree objects for X and Y to count neighbors in marginal spaces using the
    # maximum norm
    tree_X = cKDTree(data_X, leafsize=10)
    tree_Y = cKDTree(data_Y, leafsize=10)

    # Count neighbors within k-th nearest neighbor distance in X and Y spaces using the
    # maximum norm
    count_X = [
        len(tree_X.query_ball_point(p, r=d, p=np.inf)) - 1
        for p, d in zip(data_X, kth_distances)
    ]
    count_Y = [
        len(tree_Y.query_ball_point(p, r=d, p=np.inf)) - 1
        for p, d in zip(data_Y, kth_distances)
    ]

    # Compute mutual information using the KSG estimator formula
    N = len(data_X)
    # Compute local mutual information for each point
    local_mi = [
        digamma(k) - digamma(nx + 1) - digamma(ny + 1) + digamma(N)
        for nx, ny in zip(count_X, count_Y)
    ]

    # Compute aggregated mutual information
    mi = np.mean(local_mi)

    return local_mi, mi


def permutation_test_mi(
    data_X,
    data_Y,
    num_permutations,
    k=4,
    noise=1e-8,
    normalize=True,
    time_delay=0,
):
    r"""
    Perform a permutation test to calculate the p-value for the mutual information
    between two time series.

    Parameters
    ----------
    data_X : 1D numpy array
        Time series data for variable X.
    data_Y : 1D numpy array
        Time series data for variable Y.
    num_permutations : int
        The number of permutations to perform.
    k : int
        Number of nearest neighbors to consider.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.
    normalize : bool
        Flag to normalize the data to the [0, 1] range.
    time_delay : int
        Time delay to introduce between data_X and data_Y.

    Returns
    -------
    p_value : float
        The p-value of the observed mutual information.
    """

    # Compute observed MI
    _, observed_mi = mutual_info_ksg_alg1(
        data_X, data_Y, k, noise, normalize, time_delay
    )

    # Initialize a list to hold MI values from permuted data
    permuted_mis = []

    # Perform permutations
    for _ in range(num_permutations):
        # Shuffle one of the time series
        shuffled_data_X = np.random.permutation(data_X)

        # Compute MI for the shuffled data and store it
        _, permuted_mi = mutual_info_ksg_alg1(
            shuffled_data_X, data_Y, k, noise, normalize, time_delay
        )
        permuted_mis.append(permuted_mi)

    # Calculate the p-value: proportion of permuted MI values >= observed MI
    p_value = np.sum(np.array(permuted_mis) >= observed_mi) / num_permutations

    # Return the p-value directly
    return p_value


def transfer_entropy_ksg_E(data_X, data_Y, k=4, tau=1, u=0, dx=1, dy=1, noise=1e-8):
    r"""
    Estimate transfer entropy from X to Y using the Kraskov-Stoegbauer-Grassberger
    method.

    Parameters
    ----------
    data_X : array-like
        Observations for variable X.
    data_Y : array-like
        Observations for variable Y.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    dx : int
        Embedding dimension for X.
    dy : int
        Embedding dimension for Y.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.

    Returns
    -------
    local_te : array
        Local transfer entropy for each point.
    global_te : float
        Estimated transfer entropy from X to Y.
    """

    # Ensure data_X and data_Y are numpy arrays
    data_X = np.asarray(data_X)
    data_Y = np.asarray(data_Y)

    # Add Gaussian noise to the data if the flag is set
    if noise:
        data_X += np.random.normal(0, noise, data_X.shape)
        data_Y += np.random.normal(0, noise, data_Y.shape)

    N = len(data_X)
    max_delay = max(
        dy * tau + u, dx * tau
    )  # maximum delay to account for all embeddings

    # Adjusted construction of multivariate data arrays with u parameter
    joint_space_data = np.column_stack(
        (
            data_Y[max_delay + u :],  # Adjust for u in the target series
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
            np.array([data_X[i - dx * tau : i : tau] for i in range(max_delay, N - u)]),
            # Keep X embeddings aligned without u
        )
    )

    marginal_1_space_data = np.column_stack(
        (
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
            np.array([data_X[i - dx * tau : i : tau] for i in range(max_delay, N - u)]),
            # Keep X embeddings aligned without u
        )
    )

    marginal_2_space_data = np.column_stack(
        (
            data_Y[max_delay + u :],  # Adjust for u in the target series
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
        )
    )

    # Create KDTree for efficient nearest neighbor search in joint space
    tree_joint = KDTree(joint_space_data)

    # Find distances to the k-th nearest neighbor in the joint space
    distances, _ = tree_joint.query(joint_space_data, k=k + 1)
    kth_distances = distances[:, -1]

    # Count points for count_Y_present_past
    tree_Y_present_past = KDTree(marginal_2_space_data)
    count_Y_present_past = [
        len(tree_Y_present_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(marginal_2_space_data, kth_distances)
    ]

    # Count points for count_Y_past_X_past
    tree_Y_past_X_past = KDTree(marginal_1_space_data)
    count_Y_past_X_past = [
        len(tree_Y_past_X_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(marginal_1_space_data, kth_distances)
    ]

    # Count points for Count_Y_past
    data_Y_past_embedded = np.array(
        [data_Y[i - dy * tau : i : tau] for i in range(max_delay, N)]
    )
    tree_Y_past = KDTree(data_Y_past_embedded)
    Count_Y_past = [
        len(tree_Y_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(data_Y_past_embedded, kth_distances)
    ]

    # Compute local transfer entropy
    local_te = (
        digamma(k)
        - digamma(np.array(count_Y_present_past) + 1)
        - digamma(np.array(count_Y_past_X_past) + 1)
        + digamma(np.array(Count_Y_past) + 1)
    )

    # Compute global transfer entropy as the mean of the local transfer entropy
    global_te = np.mean(local_te)

    return local_te, global_te


def effective_transfer_entropy_E(data_X, data_Y, k=4, tau=1, u=0, dx=1, dy=1):
    """
    Compute the effective transfer entropy from X to Y using the
    Kraskov-Stoegbauer-Grassberger method.

    Parameters
    ----------
    data_X : array-like
        Observations for variable X.
    data_Y : array-like
        Observations for variable Y.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    dx : int
        Embedding dimension for X.
    dy : int
        Embedding dimension for Y.

    Returns
    -------
    effective_te : float
        The effective transfer entropy from X to Y.
    """
    # Compute the transfer entropy for the original X -> Y
    _, te_original = transfer_entropy_ksg_E(data_X, data_Y, k, tau, u, dx, dy)

    # Create a shuffled version of X
    data_X_shuffled = np.random.permutation(data_X)

    # Compute the transfer entropy for the shuffled X -> Y
    _, te_shuffled = transfer_entropy_ksg_E(data_X_shuffled, data_Y, k, tau, u, dx, dy)

    # Calculate the effective transfer entropy
    effective_te = te_original - te_shuffled

    return effective_te


def transfer_entropy_ksg_MN(data_X, data_Y, k=4, tau=1, u=0, dx=1, dy=1, noise=1e-8):
    r"""
    Estimate transfer entropy from X to Y using the Kraskov-Stoegbauer-Grassberger
    method.

    Parameters
    ----------
    data_X : array-like
        Observations for variable X.
    data_Y : array-like
        Observations for variable Y.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    dx : int
        Embedding dimension for X.
    dy : int
        Embedding dimension for Y.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.

    Returns
    -------
    local_te : array
        Local transfer entropy for each point.
    global_te : float
        Estimated transfer entropy from X to Y.
    """

    # Ensure data_X and data_Y are numpy arrays
    data_X = np.asarray(data_X)
    data_Y = np.asarray(data_Y)

    # Add Gaussian noise to the data if the flag is set
    if noise:
        data_X += np.random.normal(0, noise, data_X.shape)
        data_Y += np.random.normal(0, noise, data_Y.shape)

    N = len(data_X)
    max_delay = max(
        dy * tau + u, dx * tau
    )  # maximum delay to account for all embeddings

    # Adjusted construction of multivariate data arrays with u parameter
    joint_space_data = np.column_stack(
        (
            data_Y[max_delay + u :],  # Adjust for u in the target series
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
            np.array([data_X[i - dx * tau : i : tau] for i in range(max_delay, N - u)]),
            # Keep X embeddings aligned without u
        )
    )

    marginal_1_space_data = np.column_stack(
        (
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
            np.array([data_X[i - dx * tau : i : tau] for i in range(max_delay, N - u)]),
            # Keep X embeddings aligned without u
        )
    )

    marginal_2_space_data = np.column_stack(
        (
            data_Y[max_delay + u :],  # Adjust for u in the target series
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
        )
    )

    # Create cKDTree for efficient nearest neighbor search in joint space
    tree_joint = cKDTree(joint_space_data)

    # Find distances to the k-th nearest neighbor in the joint space using maximum norm
    distances, _ = tree_joint.query(joint_space_data, k=k + 1, p=np.inf)
    kth_distances = distances[:, -1]

    # Count points for count_Y_present_past
    tree_Y_present_past = KDTree(marginal_2_space_data)
    count_Y_present_past = [
        len(tree_Y_present_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(marginal_2_space_data, kth_distances)
    ]

    # Count points for count_Y_past_X_past
    tree_Y_past_X_past = KDTree(marginal_1_space_data)
    count_Y_past_X_past = [
        len(tree_Y_past_X_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(marginal_1_space_data, kth_distances)
    ]

    # Count points for Count_Y_past
    data_Y_past_embedded = np.array(
        [data_Y[i - dy * tau : i : tau] for i in range(max_delay, N)]
    )
    tree_Y_past = KDTree(data_Y_past_embedded)
    Count_Y_past = [
        len(tree_Y_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(data_Y_past_embedded, kth_distances)
    ]

    # Compute local transfer entropy
    local_te = (
        digamma(k)
        - digamma(np.array(count_Y_present_past) + 1)
        - digamma(np.array(count_Y_past_X_past) + 1)
        + digamma(np.array(Count_Y_past) + 1)
    )

    # Compute global transfer entropy as the mean of the local transfer entropy
    global_te = np.mean(local_te)

    return local_te, global_te


# Function to compute the effective transfer entropy
def effective_transfer_entropy_MN(
    data_X, data_Y, k=4, tau=1, u=0, dx=1, dy=1, noise=1e-8
):
    r"""
    Compute the effective transfer entropy from X to Y using the
    Kraskov-Stoegbauer-Grassberger method.


    Parameters
    ----------
    data_X : array-like
        Observations for variable X.
    data_Y : array-like
        Observations for variable Y.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    dx : int
        Embedding dimension for X.
    dy : int
        Embedding dimension for Y.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.

    Returns
    -------
    effective_te : float
        The effective transfer entropy from X to Y.
    """

    # Compute the transfer entropy for the original X -> Y
    _, te_original = transfer_entropy_ksg_MN(data_X, data_Y, k, tau, u, dx, dy, noise)

    # Create a shuffled version of X
    data_X_shuffled = np.random.permutation(data_X)

    # Compute the transfer entropy for the shuffled X -> Y
    _, te_shuffled = transfer_entropy_ksg_MN(data_X_shuffled, data_Y, k, tau, u, dx, dy)

    # Calculate the effective transfer entropy
    effective_te = te_original - te_shuffled

    return effective_te


def transfer_entropy_ksg(data_X, data_Y, k=4, tau=1, u=0, dx=1, dy=1, noise=1e-8):
    r"""
    Estimate transfer entropy from X to Y using the Kraskov-Stoegbauer-Grassberger
    method.

    Parameters
    ----------
    data_X : array-like
        Observations for variable X.
    data_Y : array-like
        Observations for variable Y.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    dx : int
        Embedding dimension for X.
    dy : int
        Embedding dimension for Y.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.

    Returns
    -------
    local_te : array
        Local transfer entropy for each point.
    global_te : float
        Estimated transfer entropy from X to Y.
    """

    # Ensure data_X and data_Y are numpy arrays
    data_X = np.asarray(data_X)
    data_Y = np.asarray(data_Y)

    # Add Gaussian noise to the data if the flag is set
    if noise:
        data_X += np.random.normal(0, noise, data_X.shape)
        data_Y += np.random.normal(0, noise, data_Y.shape)

    N = len(data_X)
    max_delay = max(
        dy * tau + u, dx * tau
    )  # maximum delay to account for all embeddings

    # Adjusted construction of multivariate data arrays with u parameter
    joint_space_data = np.column_stack(
        (
            data_Y[max_delay + u :],  # Adjust for u in the target series
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
            np.array([data_X[i - dx * tau : i : tau] for i in range(max_delay, N - u)]),
            # Keep X embeddings aligned without u
        )
    )

    marginal_1_space_data = np.column_stack(
        (
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
            np.array([data_X[i - dx * tau : i : tau] for i in range(max_delay, N - u)]),
            # Keep X embeddings aligned without u
        )
    )

    marginal_2_space_data = np.column_stack(
        (
            data_Y[max_delay + u :],  # Adjust for u in the target series
            np.array(
                [
                    data_Y[i - dy * tau + u : i + u : tau]
                    for i in range(max_delay, N - u)
                ]
            ),
            # Adjust embedding of Y with u
        )
    )

    # Create cKDTree for efficient nearest neighbor search in joint space
    tree_joint = cKDTree(joint_space_data)

    # Find distances to the k-th nearest neighbor in the joint space using maximum norm
    distances, _ = tree_joint.query(joint_space_data, k=k + 1, p=np.inf)
    kth_distances = distances[:, -1]

    # Count points for count_Y_present_past
    tree_Y_present_past = KDTree(marginal_2_space_data)
    count_Y_present_past = [
        len(tree_Y_present_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(marginal_2_space_data, kth_distances)
    ]

    # Count points for count_Y_past_X_past
    tree_Y_past_X_past = KDTree(marginal_1_space_data)
    count_Y_past_X_past = [
        len(tree_Y_past_X_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(marginal_1_space_data, kth_distances)
    ]

    # Count points for Count_Y_past
    data_Y_past_embedded = np.array(
        [data_Y[i - dy * tau : i : tau] for i in range(max_delay, N)]
    )
    tree_Y_past = KDTree(data_Y_past_embedded)
    Count_Y_past = [
        len(tree_Y_past.query_ball_point(p, r=d)) - 1
        for p, d in zip(data_Y_past_embedded, kth_distances)
    ]

    # Compute local transfer entropy
    local_te = (
        digamma(k)
        - digamma(np.array(count_Y_present_past) + 1)
        - digamma(np.array(count_Y_past_X_past) + 1)
        + digamma(np.array(Count_Y_past) + 1)
    )

    # Compute global transfer entropy as the mean of the local transfer entropy
    global_te = np.mean(local_te)

    return local_te, global_te


def permutation_test_te(
    data_X,
    data_Y,
    num_permutations,
    k=4,
    tau=1,
    u=0,
    dx=1,
    dy=1,
    noise=1e-8,
):
    r"""
    Perform a permutation test to calculate the p-value for the transfer entropy
    from X to Y.

    Parameters
    ----------
    data_X : 1D numpy array
        Time series data for variable X.
    data_Y : 1D numpy array
        Time series data for variable Y.
    num_permutations : int
        The number of permutations to perform.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    dx : int
        Embedding dimension for X.
    dy : int
        Embedding dimension for Y.
    noise : float or None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.

    Returns
    -------
    p_value : float
        The p-value of the observed transfer entropy.
    """

    # Compute observed TE
    _, observed_te = transfer_entropy_ksg(data_X, data_Y, k, tau, u, dx, dy, noise)

    # Initialize a list to hold TE values from permuted data
    permuted_tes = []

    # Perform permutations
    for _ in range(num_permutations):
        # Shuffle one of the time series
        shuffled_data_X = np.random.permutation(data_X)

        # Compute TE for the shuffled data and store it
        _, permuted_te = transfer_entropy_ksg(
            shuffled_data_X, data_Y, k, tau, u, dx, dy, noise
        )
        permuted_tes.append(permuted_te)

    # Calculate the p-value: proportion of permuted TE values >= observed TE
    p_value = np.sum(np.array(permuted_tes) >= observed_te) / num_permutations

    # Return the p-value directly
    return p_value
