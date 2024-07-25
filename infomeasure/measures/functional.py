"""Functional wrappers for information measures.

This module provides functional interfaces to calculate entropy, mutual information, and
transfer entropy.
The estimators are dynamically imported based on the estimator name provided,
saving time and memory by only importing the necessary classes.
"""

from functools import wraps

from .base import Estimator

entropy_estimators = {
    "discrete": "infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator",
    "kernel": "infomeasure.measures.entropy.kernel.KernelEntropyEstimator",
    "metric": "infomeasure.measures.entropy.kozachenko_leonenko."
    "KozachenkoLeonenkoEntropyEstimator",
    "kl": "infomeasure.measures.entropy.kozachenko_leonenko."
    "KozachenkoLeonenkoEntropyEstimator",
    "symbolic": "infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator",
    "permutation": "infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator",
}

mi_estimators = {
    "discrete": "infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator",
    "kernel": "infomeasure.measures.mutual_information.kernel.KernelMIEstimator",
    "metric": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "ksg": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "symbolic": "infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator",
    "permutation": "infomeasure.measures.mutual_information.symbolic."
    "SymbolicMIEstimator",
}

te_estimators = {
    "discrete": "infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator",
    "kernel": "infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator",
    "metric": "infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGTEEstimator",
    "ksg": "infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGTEEstimator",
}


def _dynamic_import(class_path):
    """Dynamically import a class from a module.

    Parameters
    ----------
    class_path : str
        The path to the class to import.

    Returns
    -------
    class
        The imported class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def _get_estimator(estimators, estimator_name):
    """Get the estimator class based on the estimator name.

    Parameters
    ----------
    estimators : dict
        The dictionary of available estimators.
    estimator_name : str
        The name of the estimator to get.

    Returns
    -------
    class
        The estimator class.

    Raises
    ------
    ValueError
        If the estimator is not recognized.
    """
    if (
        estimator_name is None
        or not isinstance(estimator_name, str)
        or (estimator_name.lower() not in estimators)
    ):
        available = ", ".join(estimators.keys())
        raise ValueError(
            f"Unknown estimator: {estimator_name}. Available estimators: {available}"
        )
    return _dynamic_import(estimators[estimator_name.lower()])


def _dynamic_estimator(estimators) -> callable:
    """Decorator to dynamically inject the estimator class into the function.

    This decorator is used to inject the estimator class into the function
    based on the estimator name provided in the arguments.
    The estimator class is then used to calculate the measure.

    Parameters
    ----------
    estimators : dict
        The dictionary of available estimators.
        Structure: {estimator_name: class_path}

    Returns
    -------
    function
        The decorated function
    """

    def decorator(func):
        @wraps(func)  # This decorator updates wrapper to look like func
        def wrapper(*args, **kwargs):
            estimator_name = kwargs.get("approach")
            if estimator_name is None:
                raise ValueError(
                    "Estimator name is required, choose one of: "
                    f"{', '.join(estimators.keys())}"
                )
            kwargs["EstimatorClass"] = _get_estimator(
                estimators, estimator_name
            )  # Inject EstimatorClass into kwargs
            return func(
                *args, **kwargs
            )  # Pass all arguments as they are, including modified kwargs

        return wrapper

    return decorator


@_dynamic_estimator(entropy_estimators)
def entropy(data, approach: str, *args, **kwargs):
    """Calculate the entropy using a functional interface of different estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete entropy estimator. <infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator>`
    2. ``kernel``: :func:`Kernel entropy estimator. <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>`
    3. [``metric``, ``kl``]: :func:`Kozachenko-Leonenko entropy estimator. <infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`
    4. [``symbolic``, ``permutation``]: :func:`Symbolic / Permutation entropy estimator. <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>`

    Parameters
    ----------
    data : array-like
        The data used to estimate the entropy.
    approach : str
        The name of the estimator to use.
    *args: tuple
        Additional arguments to pass to the estimator.
    **kwargs: dict
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    float
        The calculated entropy.

    Raises
    ------
    ValueError
        If the estimator is not recognized.
    """
    EstimatorClass = kwargs.pop("EstimatorClass")
    return EstimatorClass(data, *args, **kwargs).results()


@_dynamic_estimator(mi_estimators)
def mutual_information(
    data_x,
    data_y,
    approach: str,
    offset: int = 0,
    *args,
    **kwargs,
):
    """Calculate the mutual information using a functional interface of different
    estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete mutual information estimator. <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>`
    2. ``kernel``: :func:`Kernel mutual information estimator. <infomeasure.measures.mutual_information.kernel.KernelMIEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger mutual information estimator. <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
    4. [``symbolic``, ``permutation``]: :func:`Symbolic mutual information estimator. <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>`

    Parameters
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    approach : str
        The name of the estimator to use.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from X to Y.
    normalize : bool, optional
        If True, normalize the data before analysis. Default is False.
        Not available for the discrete estimator.
    *args: tuple
        Additional arguments to pass to the estimator.
    **kwargs: dict
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    float
        The calculated mutual information.

    Raises
    ------
    ValueError
        If the estimator is not recognized.
    """
    EstimatorClass = kwargs.pop("EstimatorClass")
    return EstimatorClass(data_x, data_y, *args, offset=offset, **kwargs).results()


@_dynamic_estimator(te_estimators)
def transfer_entropy(
    source,
    dest,
    approach: str,
    step_size: int = 1,
    src_hist_len: int = 1,
    dest_hist_len: int = 1,
    offset: int = 0,
    *args,
    **kwargs,
):
    """Calculate the transfer entropy using a functional interface of different estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>`
    2. ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`

    Parameters
    ----------
    source : array-like
        The source data used to estimate the transfer entropy.
    dest : array-like
        The destination data used to estimate the transfer entropy.
    approach : str
        The name of the estimator to use.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from source to destination.
    *args: tuple
        Additional arguments to pass to the estimator.
    **kwargs: dict
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    float
        The calculated transfer entropy.

    Raises
    ------
    ValueError
        If the estimator is not recognized.
    """
    EstimatorClass = kwargs.pop("EstimatorClass")
    return EstimatorClass(source, dest, *args, **kwargs).results()


def estimator(
    data=None,
    data_x=None,
    data_y=None,
    source=None,
    dest=None,
    *,  # the rest of the arguments are keyword-only
    measure: str = None,
    approach: str = None,
    step_size: int = 1,
    src_hist_len: int = 1,
    dest_hist_len: int = 1,
    offset: int = 0,
    **kwargs,
) -> Estimator:
    """Get an estimator for a specific measure.

    This function provides a simple interface to get
    an :class:`Estimator <.base.Estimator>` for a specific measure.

    Estimators available:

    1. Entropy:
        - ``discrete``: :func:`Discrete entropy estimator. <infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator>`
        - ``kernel``: :func:`Kernel entropy estimator. <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>`
        - [``metric``, ``kl``]: :func:`Kozachenko-Leonenko entropy estimator. <infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`
        - [``symbolic``, ``permutation``]: :func:`Symbolic / Permutation entropy estimator. <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>`

    2. Mutual Information:
        - ``discrete``: :func:`Discrete mutual information estimator. <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>`
        - ``kernel``: :func:`Kernel mutual information estimator. <infomeasure.measures.mutual_information.kernel.KernelMIEstimator>`
        - [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger mutual information estimator. <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
        - [``symbolic``, ``permutation``]: :func:`Symbolic mutual information estimator. <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>`

    3. Transfer Entropy:
        - ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>`
        - ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>`
        - [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`

    Parameters
    ----------
    data : array-like, optional
        Only if the measure is entropy.
    data_x, data_y : array-like, optional
        Only if the measure is mutual information.
    source, dest : array-like, optional
        Only if the measure is transfer entropy.
    measure : str
        The measure to estimate.
        Options: ``entropy``, ``mutual_information``, ``transfer_entropy``;
        aliases: ``h``, ``mi``, ``te``
    approach : str
        The name of the estimator to use.
        Find the available estimators in the docstring of this function.
    *args: tuple
        Additional arguments to pass to the estimator.
    **kwargs: dict
        Additional keyword arguments to pass to the estimator.

    Returns
    -------
    Estimator
        The estimator instance.

    Raises
    ------
    ValueError
        If the measure is not recognized.
    """
    if measure is None:
        raise ValueError("``measure`` is required.")
    elif measure.lower() in ["entropy", "h"]:
        if data is None:
            raise ValueError("``data`` is required for entropy estimation.")
        if any([data_x, data_y, source, dest]):
            raise ValueError(
                "Only ``data`` is required for entropy estimation, "
                "not ``data_x``, ``data_y``, ``source``, or ``dest``."
            )
        EstimatorClass = _get_estimator(entropy_estimators, approach)
        return EstimatorClass(data, **kwargs)
    elif measure.lower() in ["mutual_information", "mi"]:
        if data_x is None or data_y is None:
            raise ValueError(
                "``data_x`` and ``data_y`` are required for "
                "mutual information estimation."
            )
        if any([data, source, dest]):
            raise ValueError(
                "Only ``data_x`` and ``data_y`` are required for mutual information "
                "estimation, not ``data``, ``source``, or ``dest``."
            )
        EstimatorClass = _get_estimator(mi_estimators, approach)
        return EstimatorClass(data_x, data_y, offset=offset, **kwargs)
    elif measure.lower() in ["transfer_entropy", "te"]:
        if source is None or dest is None:
            raise ValueError(
                "``source`` and ``dest`` are required for transfer entropy estimation."
            )
        if any([data, data_x, data_y]):
            raise ValueError(
                "Only ``source`` and ``dest`` are required for transfer entropy "
                "estimation, not ``data``, ``data_x``, or ``data_y``."
            )
        EstimatorClass = _get_estimator(te_estimators, approach)
        return EstimatorClass(
            source,
            dest,
            step_size=step_size,
            src_hist_len=src_hist_len,
            dest_hist_len=dest_hist_len,
            offset=offset,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown measure: {measure}")
