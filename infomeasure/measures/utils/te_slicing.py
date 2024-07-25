"""Generalized data slicing method for transfer entropy estimators."""

from numpy import column_stack, arange, ndarray, array

from ...utils.config import logger


def construct_embedded_data_vectorized(
    data, tau=1, emb_dim=1, other_emb_dim: int = None
):
    """Construct embedded data matrix for a single time series.

    Parameters
    ----------
    data : array-like
        The time series data.
    tau : int
        Time delay for state space reconstruction.
    emb_dim : int
        Embedding dimension.
    other_emb_dim : int, optional
        Embedding dimension for the other variable.
        If provided, the larger of the two embedding dimensions is used,
        so that the resulting matrix has the same number of columns.
    """
    n = len(data)
    num_rows = (
        n
        - ((emb_dim if other_emb_dim is None else max(emb_dim, other_emb_dim)) - 1)
        * tau
    )
    if num_rows <= 0:
        return array([])

    initial_indices = arange(num_rows)
    offset_indices = arange(0, emb_dim * tau, tau)
    all_indices = initial_indices[:, None] + offset_indices

    return data[all_indices]


def te_observations(
    source,
    destination,
    src_hist_len=1,
    dest_hist_len=1,
    tau=1,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    r"""
    Slice the data arrays to prepare for TE calculation.

    For TE there are four observations that are required to calculate the transfer entropy.

    .. math::

            \hat{T}(X_{t+1}|X^{(k)}, Y^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{x}_{i+1}, x_i^{(k)}, y_i^{(l)}) g(\hat x_i^{(k)})}{g(x_i^{(k)}, y_i^{(l)}) g(\hat{x}_{i+1}, x_i^{(k)})}

    Parameters
    ----------
    source : array
        A numpy array of data points for the source variable.
    destination : array
        A numpy array of data points for the destination variable.
    src_hist_len : int, optional
        Number of past observations to consider for the source data.
        Default is 1, no history.
    dest_hist_len : int, optional
        Number of past observations to consider for the destination data.
        Default is 1, no history.
    tau : int, optional
        Step size for the time delay in the embedding. Default is 1, no delay.

    Returns
    -------
    joint_space_data : array, shape (max_len, src_hist_len + dest_hist_len + 1)
        :math:`g(\hat{x}_{i+1}, x_i^{(k)}, y_i^{(l)})`: Joint space data.
    dest_past_embedded : array, shape (max_len, dest_hist_len)
        :math:`g(\hat x_i^{(k)})` : Destination data.
    marginal_1_space_data : array, shape (max_len, dest_hist_len + src_hist_len)
        :math:`g(x_i^{(k)}, y_i^{(l)})` : Marginal space data for source and destination.
    marginal_2_space_data : array, shape (max_len, dest_hist_len + 1)
        :math:`g(\hat{x}_{i+1}, x_i^{(k)})` : Marginal space data for destination.


    With ``max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * tau``.

    Raises
    ------
    ValueError
        If the history (``src_hist_len`` or ``dest_hist_len`` times ``tau``) is greater
        than the length of the data.
    """
    # log warning if tau is >1 while src_hist_len or dest_hist_len are both 1
    if tau > 1 and src_hist_len == 1 and dest_hist_len == 1:
        logger.warning(
            "If both ``src_hist_len`` and ``dest_hist_len`` are 1, "
            "having ``tau`` > 1 does not impact the TE calculation."
        )
    # max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * tau
    # max_len cannot be negative
    if len(source) - (max(src_hist_len, dest_hist_len) - 1) * tau <= 0:
        raise ValueError(
            "The history demanded by the source and destination data "
            "is greater than the length of the data. "
            f"Demand: {(max(src_hist_len, dest_hist_len) - 1) * tau + 1}. "
            f"Data length: {len(source)}."
        )
    # Prepare multivariate data arrays

    # y_i^{(l)}
    sliced_source = construct_embedded_data_vectorized(
        source, tau=tau, emb_dim=src_hist_len, other_emb_dim=dest_hist_len
    )
    # x_i^{(k)}
    sliced_dest = construct_embedded_data_vectorized(
        destination, tau=tau, emb_dim=dest_hist_len, other_emb_dim=src_hist_len
    )

    future_dest = destination[-len(sliced_dest) :]

    # g(\hat{x}_{i+1}, x_i^{(k)}, y_i^{(l)})
    joint_space_data = column_stack((future_dest, sliced_dest, sliced_source))
    # g(\hat{x}_i^{(k)})
    # dest_past_embedded = sliced_dest
    # g(x_i^{(k)}, y_i^{(l)})
    marginal_1_space_data = column_stack((sliced_dest, sliced_source))
    # g(\hat{x}_{i+1}, x_i^{(k)})
    marginal_2_space_data = column_stack((future_dest, sliced_dest))

    return (
        joint_space_data,
        sliced_dest,
        marginal_1_space_data,
        marginal_2_space_data,
    )
