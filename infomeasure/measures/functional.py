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
    "renyi": "infomeasure.measures.entropy.renyi.RenyiEntropyEstimator",
    "symbolic": "infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator",
    "permutation": "infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator",
    "tsallis": "infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator",
}

mi_estimators = {
    "discrete": "infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator",
    "kernel": "infomeasure.measures.mutual_information.kernel.KernelMIEstimator",
    "metric": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "ksg": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "renyi": "infomeasure.measures.mutual_information.renyi.RenyiMIEstimator",
    "tsallis": "infomeasure.measures.mutual_information.tsallis.TsallisMIEstimator",
    "symbolic": "infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator",
    "permutation": "infomeasure.measures.mutual_information.symbolic."
    "SymbolicMIEstimator",
}

cmi_estimators = {
    "discrete": "infomeasure.measures.mutual_information.discrete.DiscreteCMIEstimator",
    "kernel": "infomeasure.measures.mutual_information.kernel.KernelCMIEstimator",
    "metric": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGCMIEstimator",
    "ksg": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGCMIEstimator",
    "renyi": "infomeasure.measures.mutual_information.renyi.RenyiCMIEstimator",
    "tsallis": "infomeasure.measures.mutual_information.tsallis.TsallisCMIEstimator",
    "symbolic": "infomeasure.measures.mutual_information.symbolic.SymbolicCMIEstimator",
    "permutation": "infomeasure.measures.mutual_information.symbolic."
    "SymbolicCMIEstimator",
}

te_estimators = {
    "discrete": "infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator",
    "kernel": "infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator",
    "metric": "infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGTEEstimator",
    "ksg": "infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGTEEstimator",
    "renyi": "infomeasure.measures.transfer_entropy.renyi.RenyiTEEstimator",
    "symbolic": "infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator",
    "permutation": "infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator",
    "tsallis": "infomeasure.measures.transfer_entropy.tsallis.TsallisTEEstimator",
}

cte_estimators = {
    "discrete": "infomeasure.measures.transfer_entropy.discrete.DiscreteCTEEstimator",
    "kernel": "infomeasure.measures.transfer_entropy.kernel.KernelCTEEstimator",
    "metric": "infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGCTEEstimator",
    "ksg": "infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGCTEEstimator",
    "renyi": "infomeasure.measures.transfer_entropy.renyi.RenyiCTEEstimator",
    "symbolic": "infomeasure.measures.transfer_entropy.symbolic.SymbolicCTEEstimator",
    "permutation": "infomeasure.measures.transfer_entropy.symbolic."
    "SymbolicCTEEstimator",
    "tsallis": "infomeasure.measures.transfer_entropy.tsallis.TsallisCTEEstimator",
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
    estimators : dict | [dict, dict]
        The dictionary of available estimators.
        Structure: {estimator_name: class_path}
        Or two of these, one normal and one for conditional estimators.

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
                    "Estimator name is required, choose one of: " ", ".join(
                        estimators.keys()
                        if isinstance(estimators, dict)
                        else estimators[0].keys()
                    )
                )
            # if `data_z` or `cond` is passed, it is a conditional estimator
            if isinstance(estimators, dict):
                kwargs["EstimatorClass"] = _get_estimator(
                    estimators, estimator_name
                )  # Inject EstimatorClass into kwargs
            elif kwargs.get("data_z") is not None or kwargs.get("cond") is not None:
                kwargs["EstimatorClass"] = _get_estimator(estimators[1], estimator_name)
            else:
                kwargs["EstimatorClass"] = _get_estimator(estimators[0], estimator_name)
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
    4. ``renyi``: :func:`Renyi entropy estimator. <infomeasure.measures.entropy.renyi.RenyiEntropyEstimator>`
    5. [``symbolic``, ``permutation``]: :func:`Symbolic / Permutation entropy estimator. <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>`
    6. ``tsallis``: :func:`Tsallis entropy estimator. <infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator>`

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


@_dynamic_estimator([mi_estimators, cmi_estimators])
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
    4. ``renyi``: :func:`Renyi mutual information estimator. <infomeasure.measures.mutual_information.renyi.RenyiMIEstimator>`
    5. [``symbolic``, ``permutation``]: :func:`Symbolic mutual information estimator. <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>`
    6. ``tsallis``: :func:`Tsallis mutual information estimator. <infomeasure.measures.mutual_information.tsallis.TsallisMIEstimator>`

    Parameters
    ----------
    data_x, data_y : array-like
        The data used to estimate the (conditional) mutual information.
    data_z : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    approach : str
        The name of the estimator to use.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from X to Y.
        Not compatible with the ``data_z`` parameter / conditional MI.
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


@_dynamic_estimator([te_estimators, cte_estimators])
def transfer_entropy(
    source,
    dest,
    approach: str,
    *args,
    **kwargs,
):
    """Calculate the transfer entropy using a functional interface of different estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>`
    2. ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`
    4. ``renyi``: :func:`Renyi transfer entropy estimator. <infomeasure.measures.transfer_entropy.renyi.RenyiTEEstimator>`
    5. [``symbolic``, ``permutation``]: :func:`Symbolic transfer entropy estimator. <infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator>`
    6. ``tsallis``: :func:`Tsallis transfer entropy estimator. <infomeasure.measures.transfer_entropy.tsallis.TsallisTEEstimator>`

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
        Not compatible with the ``cond`` parameter / conditional TE.
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
    prop_time: int = 0,
    src_hist_len: int = 1,
    dest_hist_len: int = 1,
    offset: int = None,
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
        - ``renyi``: :func:`Renyi entropy estimator. <infomeasure.measures.entropy.renyi.RenyiEntropyEstimator>`
        - [``symbolic``, ``permutation``]: :func:`Symbolic / Permutation entropy estimator. <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>`
        - ``tsallis``: :func:`Tsallis entropy estimator. <infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator>`

    2. Mutual Information:
        - ``discrete``: :func:`Discrete mutual information estimator. <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>`
        - ``kernel``: :func:`Kernel mutual information estimator. <infomeasure.measures.mutual_information.kernel.KernelMIEstimator>`
        - [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger mutual information estimator. <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
        - ``renyi``: :func:`Renyi mutual information estimator. <infomeasure.measures.mutual_information.renyi.RenyiMIEstimator>`
        - [``symbolic``, ``permutation``]: :func:`Symbolic mutual information estimator. <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>`
        - ``tsallis``: :func:`Tsallis mutual information estimator. <infomeasure.measures.mutual_information.tsallis.TsallisMIEstimator>`

    3. Transfer Entropy:
        - ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>`
        - ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>`
        - [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`
        - ``renyi``: :func:`Renyi transfer entropy estimator. <infomeasure.measures.transfer_entropy.renyi.RenyiTEEstimator>`
        - [``symbolic``, ``permutation``]: :func:`Symbolic transfer entropy estimator. <infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator>`
        - ``tsallis``: :func:`Tsallis transfer entropy estimator. <infomeasure.measures.transfer_entropy.tsallis.TsallisTEEstimator>`

    Parameters
    ----------
    data : array-like, optional
        Only if the measure is entropy.
    data_x, data_y : array-like, optional
        Only if the measure is mutual information.
    source, dest : array-like, optional
        Only if the measure is transfer entropy.
    data_z : array-like, optional
        Only if the measure is conditional mutual information.
    cond : array-like, optional
        Only if the measure is conditional transfer entropy.
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
        if any(
            [data_x, data_y, source, dest, kwargs.get("data_z"), kwargs.get("cond")]
        ):
            raise ValueError(
                "Only ``data`` is required for entropy estimation, "
                "not ``data_x``, ``data_y``, ``source``, ``dest``, "
                "``data_z``, or ``cond``."
            )
        EstimatorClass = _get_estimator(entropy_estimators, approach)
        return EstimatorClass(data, **kwargs)
    elif measure.lower() in ["mutual_information", "mi"]:
        if data_x is None or data_y is None:
            raise ValueError(
                "``data_x`` and ``data_y`` are required for "
                "mutual information estimation."
            )
        if any([data, source, dest, kwargs.get("cond")]):
            raise ValueError(
                "Only ``data_x`` and ``data_y`` are required for mutual information "
                "estimation, not ``data``, ``source``, ``dest``, or ``cond``."
            )
        if kwargs.get("data_z") is not None:
            EstimatorClass = _get_estimator(cmi_estimators, approach)
        else:
            EstimatorClass = _get_estimator(mi_estimators, approach)
        return EstimatorClass(data_x, data_y, offset=offset, **kwargs)
    elif measure.lower() in ["transfer_entropy", "te"]:
        if source is None or dest is None:
            raise ValueError(
                "``source`` and ``dest`` are required for transfer entropy estimation."
            )
        if any([data, data_x, data_y, kwargs.get("data_z")]):
            raise ValueError(
                "Only ``source`` and ``dest`` are required for transfer entropy "
                "estimation, not ``data``, ``data_x``, ``data_y``, or ``data_z``."
            )
        if kwargs.get("cond"):
            EstimatorClass = _get_estimator(cte_estimators, approach)
        else:
            EstimatorClass = _get_estimator(te_estimators, approach)
        return EstimatorClass(
            source,
            dest,
            prop_time=prop_time,
            src_hist_len=src_hist_len,
            dest_hist_len=dest_hist_len,
            step_size=step_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown measure: {measure}")
