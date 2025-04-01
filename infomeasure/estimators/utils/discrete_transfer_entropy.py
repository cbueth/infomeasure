"""Functions for efficient computation of discrete transfer entropy."""

from numpy import log, ndarray

from .discrete_interaction_information import (
    conditional_mutual_information_global,
    conditional_mutual_information_local,
)
from .ordinal import reduce_joint_space


def combined_te_form(
    slice_method, *data, local: bool = False, log_func: callable = log, **slice_kwargs
) -> float | ndarray:
    """
    Calculate the Transfer Entropy using the combined TE formula.

    Parameters
    ----------
    slice_method : function
        The slicing method to use for the symbolized data.
    *data : array-like
        The source, destination, and if applicable, conditional data.
    local : bool, optional
        Whether to calculate the local transfer entropy.
        If False, the global transfer entropy is calculated.
        Default is False.
    log_func : callable, optional
        The logarithm function to use. Default is the natural logarithm.
    **slice_kwargs : dict
        The history lengths for the source, destination, and if applicable,
        conditional data.

    Returns
    -------
    float
        The Transfer Entropy value.
    """
    cmi_func = (
        conditional_mutual_information_local
        if local
        else conditional_mutual_information_global
    )
    sliced_data = slice_method(
        *data,
        **slice_kwargs,
        construct_joint_spaces=False,
    )
    if len(sliced_data) == 3:
        src_history, dest_history, dest_future = sliced_data
        return cmi_func(
            dest_future,
            src_history,
            cond=reduce_joint_space(dest_history),
            log_func=log_func,
        )
    elif len(sliced_data) == 4:
        src_history, dest_history, dest_future, cond_history = sliced_data
        return cmi_func(
            dest_future,
            src_history,
            cond=reduce_joint_space((dest_history, cond_history)),
            log_func=log_func,
        )
    else:
        raise ValueError(
            "Invalid number of data arrays. "
            "The slice method returned an invalid number of sliced data."
        )
