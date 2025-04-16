r"""Generalized data slicing method for transfer entropy estimators.

This module provides a method to slice the data arrays to prepare for transfer
entropy (TE) calculation.
The TE measures the information flow from a source variable (X) to
a target/destination variable (Y).
In this context, the future state is always associated with
the target/destination variable.

Conventions:

- ``X``: Source variable
- ``Y``: Destination/target variable
- ``dest_future``: Future state of the destination variable (Y)
- ``dest_history``: Past states of the destination variable (Y)
- ``src_history``: Past states of the source variable (X)

The TE is calculated as:

.. math::

    \hat{T}(Y_{t+1}|Y^{(k)}, X^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{y}_{i+1}, y_i^{(k)}, x_i^{(l)}) g(\hat y_i^{(k)})}{g(y_i^{(k)}, x_i^{(l)}) g(\hat{y}_{i+1}, y_i^{(k)})}

"""

from typing import Iterable

from numpy import arange, ndarray, concatenate, expand_dims, issubdtype, integer
from numpy.random import Generator, default_rng

from ...utils.config import logger


def te_observations(
    source,
    destination,
    src_hist_len=1,
    dest_hist_len=1,
    step_size=1,
    permute_src=False,
    resample_src=False,
    construct_joint_spaces: bool = True,
) -> tuple[ndarray, ndarray, ndarray, ndarray] | Iterable | tuple:
    r"""
    Slice the data arrays to prepare for TE calculation.

    For TE there are four observations that are required to calculate the
    transfer entropy.

    .. math::

            \hat{T}(Y_{t+1}|Y^{(k)}, X^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{y}_{i+1}, y_i^{(k)}, x_i^{(l)}) g(\hat y_i^{(k)})}{g(y_i^{(k)}, x_i^{(l)}) g(\hat{y}_{i+1}, y_i^{(k)})}

    Parameters
    ----------
    source : array, shape (n,)
        A numpy array of data points for the source variable (X).
    destination : array, shape (n,)
        A numpy array of data points for the destination variable (Y).
    src_hist_len : int, optional
        Number of past observations (l) to consider for the source data (X).
        Default is 1, only one current observation, no further history.
        One future observation is always considered for the source data.
    dest_hist_len : int, optional
        Number of past observations (k) to consider for the destination data (Y).
        Default is 1, only one current observation, no further history.
    step_size : int, optional
        Step size for the time delay in the embedding.
        Default is None, which equals to 1, every observation is considered.
        If step_size is greater than 1, the history is subsampled.
        This applies to both the source and destination data.
    permute_src : bool | Generator, optional
        Whether to shuffle the sliced source history data. Default is False.
        This is used for the permutation TE. Rows are permuted, keeping the
        history intact.
        If a random number generator is provided, it will be used for shuffling.
        If True, a new random number generator will be created.
    resample_src : bool | Generator, optional
        Whether to resample the sliced source history data. Default is False.
        This is used for the permutation TE using bootstrapping.
        Rows are resampled with replacement, keeping the history intact.
        If a random number generator is provided, it will be used for resampling.
        If True, a new random number generator will be created.
    construct_joint_spaces : bool, optional
        Whether to construct the joint spaces. Default is True.
        If False, the sliced source and destination data are returned instead.

    Returns
    -------
    joint_space_data : array, shape (max_len, src_hist_len + dest_hist_len + 1)
        :math:`g(x_i^{(l)}, y_i^{(k)}, \hat{y}_{i+1})`: Joint space data.
    dest_past_embedded : array, shape (max_len, dest_hist_len)
        :math:`g(\hat y_i^{(k)})` : Embedded past destination data.
    marginal_1_space_data : array, shape (max_len, dest_hist_len + src_hist_len)
        :math:`g(x_i^{(l)}, y_i^{(k)})` : Marginal space data for destination and
        source.
    marginal_2_space_data : array, shape (max_len, dest_hist_len + 1)
        :math:`g(y_i^{(k)}, \hat{y}_{i+1})` : Marginal space data for destination.
    sliced data : tuple of arrays
        If ``construct_joint_spaces`` is False, the sliced source and destination
        data are returned instead. Namely, the tuple contains:

        - ``src_history`` : array, shape (max_len, src_hist_len)
          :math:`x_i^{(l)}` : Source history.
        - ``dest_history`` : array, shape (max_len, dest_hist_len)
          :math:`y_i^{(k)}` : Destination history.
        - ``dest_future`` : array, shape (max_len,)
          :math:`\hat{y}_{i+1}` : Destination future.

    Notes
    -----
    - For permutation TE, ``permute_src`` xor ``resample_src`` can be used.
    - With ``max_len = data_len - (max(src_hist_len, dest_hist_len) - 1) * step_size``.

    Raises
    ------
    ValueError
        If the history (``src_hist_len`` or ``dest_hist_len`` times ``step_size``) is
        greater than the length of the data.
    ValueError
        If ``src_hist_len``, ``dest_hist_len``, or ``step_size`` are
        not positive integers.
    ValueError
        If both ``permute_src`` and ``resample_src`` are provided.
    """
    # log warning if step_size is >1 while src_hist_len or dest_hist_len are both 1
    if step_size > 1 and src_hist_len == 1 and dest_hist_len == 1:
        logger.warning(
            "If both ``src_hist_len`` and ``dest_hist_len`` are 1, "
            "having ``step_size`` > 1 does not impact the TE calculation."
        )
    # error if vars are not positive integers
    if not all(
        issubdtype(type(var), integer) and var > 0
        for var in (src_hist_len, dest_hist_len, step_size)
    ):
        raise ValueError(
            "src_hist_len, dest_hist_len, and step_size must be positive integers."
        )

    max_delay = max(dest_hist_len, src_hist_len) * step_size
    # max delay must be less than the length of the data, otherwise raise an error
    if max_delay >= len(source):
        raise ValueError(
            "The history demanded by the source and destination data "
            "is greater than the length of the data and results in empty arrays."
        )

    if permute_src and resample_src:
        raise ValueError("Only one of permute_src or resample_src can be provided.")

    base_indices = arange(max_delay, len(destination), step_size)

    # Construct src_history
    offset_indices = arange(step_size, (src_hist_len + 1) * step_size, step_size)
    src_history_indices = base_indices[:, None] - offset_indices[::-1]
    if isinstance(permute_src, Generator):
        permute_src.shuffle(src_history_indices, axis=0)  # in-place
    elif permute_src:
        rng = default_rng()
        rng.shuffle(src_history_indices, axis=0)  # in-place
    if isinstance(resample_src, Generator):
        src_history_indices = resample_src.choice(  # re-assign
            src_history_indices, size=src_history_indices.shape[0], axis=0, replace=True
        )
    elif resample_src:
        rng = default_rng()
        src_history_indices = rng.choice(  # re-assign
            src_history_indices, size=src_history_indices.shape[0], axis=0, replace=True
        )
    # get the data using the indices
    src_history = source[src_history_indices]

    # Construct dest_history
    offset_indices = arange(step_size, (dest_hist_len + 1) * step_size, step_size)
    dest_history_indices = base_indices[:, None] - offset_indices[::-1]
    dest_history = destination[dest_history_indices]

    # Construct dest_future
    dest_future = destination[base_indices]
    # src_future: (data_len,) -> (data_len, 1); or (data_len, m) -> (data_len, 1, m)
    dest_future = expand_dims(dest_future, axis=1)

    # flatten into two dimensions if the array has more than two dimensions
    src_history = (
        src_history
        if src_history.ndim < 3
        else src_history.reshape(src_history.shape[0], -1)
    )
    dest_history = (
        dest_history
        if dest_history.ndim < 3
        else dest_history.reshape(dest_history.shape[0], -1)
    )
    dest_future = (
        dest_future
        if dest_future.ndim < 3
        else dest_future.reshape(dest_future.shape[0], -1)
    )
    if construct_joint_spaces:
        return _construct_joint_space_data(
            src_history,  # x_i^{(l)}
            dest_history,  # y_i^{(k)}
            dest_future,  # \hat{y}_{i+1}
        )
    else:
        return src_history, dest_history, dest_future


def cte_observations(
    source,
    destination,
    condition,
    src_hist_len=1,
    dest_hist_len=1,
    cond_hist_len=1,
    step_size=1,
    construct_joint_spaces: bool = True,
) -> tuple[ndarray, ndarray, ndarray, ndarray] | Iterable | tuple:
    r"""
    Slice the data arrays to prepare for CTE calculation.

    For CTE there are four observations that are required to calculate the
    conditional transfer entropy.

    .. math::

            \hat{T}(Y_{t+1}|Y^{(k)}, X^{(l)}) = \frac{1}{N} \sum_{i=1}^{N} \log \frac{g(\hat{y}_{i+1}, y_i^{(k)}, z_i^{(m)}, x_i^{(l)}) g(\hat y_i^{(k)}, z_i^{(m)})}{g(y_i^{(k)}, z_i^{(m)}, x_i^{(l)}) g(\hat{y}_{i+1}, y_i^{(k)}, z_i^{(m)})}

    Parameters
    ----------
    source : array, shape (n,)
        A numpy array of data points for the source variable (X).
    destination : array, shape (n,)
        A numpy array of data points for the destination variable (Y).
    condition : array, shape (n,)
        A numpy array of data points for the conditioning variable (Z).
    src_hist_len : int, optional
        Number of past observations (l) to consider for the source data (X).
        Default is 1, only one current observation, no further history.
        One future observation is always considered for the source data.
    dest_hist_len : int, optional
        Number of past observations (k) to consider for the destination data (Y).
        Default is 1, only one current observation, no further history.
    cond_hist_len : int, optional
        Number of past observations (m) to consider for the conditioning data (Z).
        Default is 1, only one current observation, no further history.
    step_size : int, optional
        Step size for the time delay in the embedding.
        Default is None, which equals to 1, every observation is considered.
        If step_size is greater than 1, the history is subsampled.
        This applies to both the source and destination data.
    construct_joint_spaces : bool, optional
        Whether to construct the joint spaces. Default is True.
        If False, the sliced source and destination data are returned instead.

    Returns
    -------
    joint_space_data : array, shape (max_len, src_hist_len + dest_hist_len + cond_hist_len + 1)
        :math:`g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)}, \hat{y}_{i+1})`:
        Conditional joint space data.
    dest_past_embedded : array, shape (max_len, dest_hist_len)
        :math:`g(\hat y_i^{(k)}, z_i^{(m)})` :
        Conditional embedded past destination data.
    marginal_1_space_data : array, shape (max_len, dest_hist_len + src_hist_len)
        :math:`g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)})` :
        Conditional marginal space data for destination and source.
    marginal_2_space_data : array, shape (max_len, dest_hist_len + 1)
        :math:`g(z_i^{(m)}, y_i^{(k)}, \hat{y}_{i+1})` :
        Conditional marginal space data for destination.
    sliced data : tuple of arrays
        If ``construct_joint_spaces`` is False, the sliced source, destination and
        conditional data are returned instead.
        Namely, the tuple contains:

        - ``src_history`` : array, shape (max_len, src_hist_len)
          :math:`x_i^{(l)}` : Source history.
        - ``dest_history`` : array, shape (max_len, dest_hist_len)
          :math:`y_i^{(k)}` : Destination history.
        - ``dest_future`` : array, shape (max_len,)
          :math:`\hat{y}_{i+1}` : Destination future.
        - ``cond_history`` : array, shape (max_len, cond_hist_len)
          :math:`z_i^{(m)}` : Condition history.

    Notes
    -----
    With ``max_len = data_len - (max(src_hist_len, dest_hist_len, cond_hist_len) - 1)
    * step_size``.

    Raises
    ------
    TypeError
        If the arguments are wrong types.
    ValueError
        If the history (``src_hist_len`` or ``dest_hist_len`` or ``cond_hist_len`` times
        ``step_size``) is greater than the length of the data.
    ValueError
        If ``src_hist_len``, ``dest_hist_len``, ``cond_hist_len``, or ``step_size`` are
        not positive integers.
    """
    if not all(
        issubdtype(type(var), integer) and var > 0
        for var in (src_hist_len, dest_hist_len, cond_hist_len, step_size)
    ):
        raise ValueError(
            "src_hist_len, dest_hist_len, cond_hist_len, and step_size "
            "must be positive integers."
        )
    if not all(isinstance(var, ndarray) for var in (source, destination, condition)):
        raise TypeError("source, destination, and condition must be numpy arrays.")
    # log warning if step_size is >1 while src_hist_len or dest_hist_len are both 1
    if (
        step_size > 1
        and src_hist_len == 1
        and dest_hist_len == 1
        and cond_hist_len == 1
    ):
        logger.warning(
            "If all ``src_hist_len``, ``dest_hist_len``, and ``cond_hist_len`` are 1, "
            "having ``step_size`` > 1 does not impact the TE calculation."
        )
    # error if vars are not positive integers
    if not all(
        issubdtype(type(var), integer) and var > 0
        for var in (src_hist_len, dest_hist_len, cond_hist_len, step_size)
    ):
        raise ValueError(
            "src_hist_len, dest_hist_len, cond_hist_len, and step_size "
            "must be positive integers."
        )

    max_delay = max(dest_hist_len, src_hist_len, cond_hist_len) * step_size
    # max delay must be less than the length of the data, otherwise raise an error
    if max_delay >= len(source):
        raise ValueError(
            "The history demanded by the source, destination, and condition data "
            "is greater than the length of the data and results in empty arrays."
        )

    base_indices = arange(max_delay, len(destination), step_size)

    # Construct src_history
    offset_indices = arange(step_size, (src_hist_len + 1) * step_size, step_size)
    src_history_indices = base_indices[:, None] - offset_indices[::-1]
    src_history = source[src_history_indices]

    # Construct cond_history
    offset_indices = arange(step_size, (cond_hist_len + 1) * step_size, step_size)
    cond_history_indices = base_indices[:, None] - offset_indices[::-1]
    cond_history = condition[cond_history_indices]

    # Construct dest_history
    offset_indices = arange(step_size, (dest_hist_len + 1) * step_size, step_size)
    dest_history_indices = base_indices[:, None] - offset_indices[::-1]
    dest_history = destination[dest_history_indices]

    # Construct dest_future
    dest_future = destination[base_indices]
    # src_future: (data_len,) -> (data_len, 1); or (data_len, m) -> (data_len, 1, m)
    dest_future = expand_dims(dest_future, axis=1)

    # flatten into two dimensions if the array has more than two dimensions
    src_history = (
        src_history
        if src_history.ndim < 3
        else src_history.reshape(src_history.shape[0], -1)
    )
    cond_history = (
        cond_history
        if cond_history.ndim < 3
        else cond_history.reshape(cond_history.shape[0], -1)
    )
    dest_history = (
        dest_history
        if dest_history.ndim < 3
        else dest_history.reshape(dest_history.shape[0], -1)
    )
    dest_future = (
        dest_future
        if dest_future.ndim < 3
        else dest_future.reshape(dest_future.shape[0], -1)
    )

    if construct_joint_spaces:
        return _construct_joint_space_data(
            src_history,  # x_i^{(l)}
            dest_history,  # y_i^{(k)}
            dest_future,  # \hat{y}_{i+1}
            cond_history,  # z_i^{(m)}
        )
    else:
        return src_history, dest_history, dest_future, cond_history


def _construct_joint_space_data(
    *sliced_data,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Construct joint space data.

    Parameters
    ----------
    src_history : np.ndarray
        Source history.
    dest_history : np.ndarray
        Destination history.
    dest_future : np.ndarray
        Destination future.
    cond_history : np.ndarray, optional
        Conditioning history.
    Returns
    -------
    tuple of np.ndarray
        Joint space data, as returned by :func:`te_observations` and
        :func:`cte_observations`, if conditioning history is provided.
    """

    # g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)}, \hat{y}_{i+1})
    joint_space_data = concatenate(
        sliced_data,  # x_i^{(l)}, y_i^{(k)}, \hat{y}_{i+1}, z_i^{(m)}
        axis=1,  # or x_i^{(l)}, y_i^{(k)}, \hat{y}_{i+1}
    )

    if len(sliced_data) == 3:  # g(\hat y_i^{(k)})
        dest_past_embedded = sliced_data[1]
    else:  # g(\hat y_i^{(k)}, z_i^{(m)})
        dest_past_embedded = concatenate((sliced_data[1], *sliced_data[3:]), axis=1)

    # g(x_i^{(l)}, y_i^{(k)}, z_i^{(m)})
    marginal_1_space_data = concatenate(
        (*sliced_data[:2], *sliced_data[3:]), axis=1
    )  # x_i^{(l)}, y_i^{(k)}, z_i^{(m)} or x_i^{(l)}, y_i^{(k)}

    # g(y_i^{(k)}, \hat{y}_{i+1}, z_i^{(m)})
    marginal_2_space_data = concatenate(
        sliced_data[1:],  # y_i^{(k)}, \hat{y}_{i+1}, z_i^{(m)}
        axis=1,  # or y_i^{(k)}, \hat{y}_{i+1}
    )

    return (
        joint_space_data,
        dest_past_embedded,
        marginal_1_space_data,
        marginal_2_space_data,
    )
