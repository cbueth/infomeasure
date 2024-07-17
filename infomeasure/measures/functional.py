"""Functional wrappers for information measures.

This module provides functional interfaces to calculate entropy, mutual information, and
transfer entropy.
The estimators are dynamically imported based on the estimator name provided,
saving time and memory by only importing the necessary classes.
"""

from functools import wraps

entropy_estimators = {
    "discrete": "infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator",
    "kernel": "infomeasure.measures.entropy.kernel.KernelEntropyEstimator",
    "metric": "infomeasure.measures.entropy.kozachenko_leonenko."
    "KozachenkoLeonenkoEntropyEstimator",
    "kl": "infomeasure.measures.entropy.kozachenko_leonenko."
    "KozachenkoLeonenkoEntropyEstimator",
}

mi_estimators = {
    "discrete": "infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator",
    "kernel": "infomeasure.measures.mutual_information.kernel.KernelMIEstimator",
    "metric": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "ksg": "infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
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
    if estimator_name.lower() not in estimators:
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
            estimator_name = kwargs.get("estimator")
            kwargs["EstimatorClass"] = _get_estimator(
                estimators, estimator_name
            )  # Inject EstimatorClass into kwargs
            return func(
                *args, **kwargs
            )  # Pass all arguments as they are, including modified kwargs

        return wrapper

    return decorator


@_dynamic_estimator(entropy_estimators)
def entropy(data, estimator: str, *args, **kwargs):
    """Calculate the entropy using a functional interface of different estimators.

    Supports the following estimators:

    1. ``discrete``: :func:`Discrete entropy estimator. <infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator>`
    2. ``kernel``: :func:`Kernel entropy estimator. <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>`
    3. [``metric``, ``kl``]: :func:`Kozachenko-Leonenko entropy estimator. <infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`

    Parameters
    ----------
    data : array-like
        The data used to estimate the entropy.
    estimator : str
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
def mutual_information(data_x, data_y, estimator: str, *args, **kwargs):
    """Calculate the mutual information using a functional interface of different estimators.

    Supports the following estimators:

    1. ``discrete``: :func:`Discrete mutual information estimator. <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>`
    2. ``kernel``: :func:`Kernel mutual information estimator. <infomeasure.measures.mutual_information.kernel.KernelMIEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger mutual information estimator. <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`

    Parameters
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    estimator : str
        The name of the estimator to use.
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
    return EstimatorClass(data_x, data_y, *args, **kwargs).results()


@_dynamic_estimator(te_estimators)
def transfer_entropy(source, dest, estimator: str, *args, **kwargs):
    """Calculate the transfer entropy using a functional interface of different estimators.

    Supports the following estimators:

    1. ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>`
    2. ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`

    Parameters
    ----------
    source : array-like
        The source data used to estimate the transfer entropy.
    dest : array-like
        The destination data used to estimate the transfer entropy.
    estimator : str
        The name of the estimator to use.
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
