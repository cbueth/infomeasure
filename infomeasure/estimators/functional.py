"""Functional wrappers for information estimators.

This module provides functional interfaces to calculate entropy, mutual information, and
transfer entropy.
The estimators are dynamically imported based on the estimator name provided,
saving time and memory by only importing the necessary classes.
"""

from functools import wraps

from ..utils.config import logger
from .base import EstimatorType

entropy_estimators = {
    "discrete": "infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator",
    "kernel": "infomeasure.estimators.entropy.kernel.KernelEntropyEstimator",
    "metric": "infomeasure.estimators.entropy.kozachenko_leonenko."
    "KozachenkoLeonenkoEntropyEstimator",
    "kl": "infomeasure.estimators.entropy.kozachenko_leonenko."
    "KozachenkoLeonenkoEntropyEstimator",
    "renyi": "infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator",
    "ordinal": "infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator",
    "symbolic": "infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator",
    "permutation": "infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator",
    "tsallis": "infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator",
}

mi_estimators = {
    "discrete": "infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator",
    "kernel": "infomeasure.estimators.mutual_information.kernel.KernelMIEstimator",
    "metric": "infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "ksg": "infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGMIEstimator",
    "renyi": "infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator",
    "tsallis": "infomeasure.estimators.mutual_information.tsallis.TsallisMIEstimator",
    "ordinal": "infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator",
    "symbolic": "infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator",
    "permutation": "infomeasure.estimators.mutual_information.ordinal."
    "OrdinalMIEstimator",
}

# for M(x), use map to H(x)
mi_entropy_map = {mi_key: mi_key for mi_key in mi_estimators.keys()}
# Overwrite shorthands that differ between MI and H
mi_entropy_map["ksg"] = "kl"

cmi_estimators = {
    "discrete": "infomeasure.estimators.mutual_information.discrete.DiscreteCMIEstimator",
    "kernel": "infomeasure.estimators.mutual_information.kernel.KernelCMIEstimator",
    "metric": "infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGCMIEstimator",
    "ksg": "infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger."
    "KSGCMIEstimator",
    "renyi": "infomeasure.estimators.mutual_information.renyi.RenyiCMIEstimator",
    "tsallis": "infomeasure.estimators.mutual_information.tsallis.TsallisCMIEstimator",
    "ordinal": "infomeasure.estimators.mutual_information.ordinal.OrdinalCMIEstimator",
    "symbolic": "infomeasure.estimators.mutual_information.ordinal.OrdinalCMIEstimator",
    "permutation": "infomeasure.estimators.mutual_information.ordinal."
    "OrdinalCMIEstimator",
}

te_estimators = {
    "discrete": "infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator",
    "kernel": "infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator",
    "metric": "infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGTEEstimator",
    "ksg": "infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGTEEstimator",
    "renyi": "infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator",
    "ordinal": "infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator",
    "symbolic": "infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator",
    "permutation": "infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator",
    "tsallis": "infomeasure.estimators.transfer_entropy.tsallis.TsallisTEEstimator",
}

cte_estimators = {
    "discrete": "infomeasure.estimators.transfer_entropy.discrete.DiscreteCTEEstimator",
    "kernel": "infomeasure.estimators.transfer_entropy.kernel.KernelCTEEstimator",
    "metric": "infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGCTEEstimator",
    "ksg": "infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger."
    "KSGCTEEstimator",
    "renyi": "infomeasure.estimators.transfer_entropy.renyi.RenyiCTEEstimator",
    "ordinal": "infomeasure.estimators.transfer_entropy.ordinal.OrdinalCTEEstimator",
    "symbolic": "infomeasure.estimators.transfer_entropy.ordinal.OrdinalCTEEstimator",
    "permutation": "infomeasure.estimators.transfer_entropy.ordinal."
    "OrdinalCTEEstimator",
    "tsallis": "infomeasure.estimators.transfer_entropy.tsallis.TsallisCTEEstimator",
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


def get_estimator_class(measure=None, approach=None) -> EstimatorType:
    """Get estimator class based on the estimator name and approach.

    This function returns the estimator class based on the measure and approach
    provided.
    If you want an instance of an estimator, initialized with data and parameters,
    use the functional interface :func:`estimator`.

    Parameters
    ----------
    measure : str
        The measure to estimate.
        Options: ``entropy``, ``mutual_information``, ``transfer_entropy``,
        ``conditional_mutual_information``, ``conditional_transfer_entropy``.
        Aliases: ``h``, ``mi``, ``te``, ``cmi``, ``cte``.
    approach : str
        The name of the estimator to use.

    Returns
    -------
    class
        The estimator class.

    Raises
    ------
    ValueError
        If the measure is not recognized.
    ValueError
        If the approach is not recognized.
    """
    if measure is None:
        raise ValueError("The measure must be specified.")
    if measure.lower() in ["entropy", "h"]:
        return _get_estimator(entropy_estimators, approach)
    elif measure.lower() in ["mutual_information", "mi"]:
        return _get_estimator(mi_estimators, approach)
    elif measure.lower() in ["conditional_mutual_information", "cmi"]:
        return _get_estimator(cmi_estimators, approach)
    elif measure.lower() in ["transfer_entropy", "te"]:
        return _get_estimator(te_estimators, approach)
    elif measure.lower() in ["conditional_transfer_entropy", "cte"]:
        return _get_estimator(cte_estimators, approach)
    else:
        raise ValueError(
            f"Unknown measure: {measure}. Available measures: entropy, mutual_information, "
            "conditional_mutual_information, transfer_entropy, conditional_transfer_entropy."
        )


def _dynamic_estimator(measure) -> callable:
    """Decorator to dynamically inject the estimator class into the function.

    This decorator is used to inject the estimator class into the function
    based on the estimator name provided in the arguments.
    The estimator class is then used to calculate the measure.

    Parameters
    ----------
    measure : dict | [dict, dict]
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
                raise ValueError("``approach`` must be provided")
            # if  `cond` is passed, it is a conditional estimator
            if isinstance(measure, str):
                # Inject EstimatorClass into kwargs
                kwargs["EstimatorClass"] = get_estimator_class(measure, estimator_name)
            elif kwargs.get("cond") is not None:
                kwargs["EstimatorClass"] = get_estimator_class(
                    measure[1], estimator_name
                )
            else:
                kwargs["EstimatorClass"] = get_estimator_class(
                    measure[0], estimator_name
                )
            return func(
                *args, **kwargs
            )  # Pass all arguments as they are, including modified kwargs

        return wrapper

    return decorator


@_dynamic_estimator("entropy")
def entropy(*data, approach: str, **kwargs: any):
    r"""Calculate the (joint) entropy using a functional interface of different estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete entropy estimator. <infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator>`
    2. ``kernel``: :func:`Kernel entropy estimator. <infomeasure.estimators.entropy.kernel.KernelEntropyEstimator>`
    3. [``metric``, ``kl``]: :func:`Kozachenko-Leonenko entropy estimator. <infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`
    4. ``renyi``: :func:`Renyi entropy estimator. <infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator>`
    5. [``ordinal``, ``symbolic``, ``permutation``]: :func:`Ordinal / Permutation entropy estimator. <infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator>`
    6. ``tsallis``: :func:`Tsallis entropy estimator. <infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator>`


    For the discrete Shannon entropy this is

    .. math::

       \texttt{im.entropy(data_X, approach="discrete")} = H(X) = -\sum_{x \in X} p(x) \log p(x).

    Where for :math:`H(x)`, the estimated pmf :math:`p(x)` belongs to the RV :math:`X`.

    .. math::

       \texttt{im.entropy(data_P, data_Q, ...)} = H_Q(P) = H_\times(P, Q)
       = -\sum_{x \in X} p(x) \log q(x)

    For the cross-entropy :math:`H_Q(P)`,
    the estimated pmf :math:`p(x)` belongs to the RV :math:`P`
    and :math:`q(x)` to the RV :math:`Q`.
    For other approaches, this formula is generalized in different forms.

    Parameters
    ----------
    *data : array-like
        The data used to estimate the entropy.
        For entropy, this can be an array-like.
        For joint entropy, pass the joint values inside a tuple.
        For cross-entropy, pass two separate parameters.
    approach : str
        The name of the estimator to use.
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
    return EstimatorClass(*data, **kwargs).result()


def cross_entropy(*data, **kwargs: any):
    """Calculate the cross-entropy using a functional interface of different
    estimators.

    See :func:`entropy` for more details on the parameters and returns.
    """
    if len(data) < 2:
        raise ValueError(
            "Cross-entropy requires at least two random variables "
            "passed as positional parameters: `cross_entropy(var1, var2, **kwargs)`"
        )
    return entropy(*data, **kwargs)


@_dynamic_estimator(["mi", "cmi"])
def mutual_information(
    *data,
    approach: str,
    **kwargs: any,
):
    """Calculate the mutual information using a functional interface of different
    estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete mutual information estimator. <infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator>`
    2. ``kernel``: :func:`Kernel mutual information estimator. <infomeasure.estimators.mutual_information.kernel.KernelMIEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger mutual information estimator. <infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
    4. ``renyi``: :func:`Renyi mutual information estimator. <infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator>`
    5. [``ordinal``, ``symbolic``, ``permutation``]: :func:`Ordinal mutual information estimator. <infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator>`
    6. ``tsallis``: :func:`Tsallis mutual information estimator. <infomeasure.estimators.mutual_information.tsallis.TsallisMIEstimator>`

    Parameters
    ----------
    *data : array-like
        The data used to estimate the (conditional) mutual information.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    approach : str
        The name of the estimator to use.
    normalize : bool, optional
        If True, normalize the data before analysis. Default is False.
        Not available for the discrete estimator.
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
    if len(data) < 2:
        raise ValueError(
            "Mutual Information requires at least two variables as arguments. "
            "If needed, a conditional variable can be passed as keyword argument: "
            "`mutual_information(data1, data2, data3, ..., **kwargs)` or "
            "`mutual_information(data1, data2, ..., cond=cond_var, **kwargs)`"
        )
    return EstimatorClass(*data, **kwargs).result()


def conditional_mutual_information(*data, **kwargs: any):
    """Conditional mutual information between two variables given a third variable.

    See :func:`mutual_information <mutual_information>` for more information.
    """
    if kwargs.get("cond") is None:
        raise ValueError(
            "CMI requires a conditional variable. Pass a 'cond' keyword argument."
        )
    if len(data) < 2:
        raise ValueError(
            "CMI requires at least two variables as arguments and "
            "a conditional variable as keyword argument: "
            "`conditional_mutual_information("
            "data1, data2, ..., cond=cond_var, **kwargs)`"
        )
    return mutual_information(*data, **kwargs)


@_dynamic_estimator(["te", "cte"])
def transfer_entropy(
    *data,
    approach: str,
    **kwargs: any,
):
    """Calculate the transfer entropy using a functional interface of different estimators.

    Supports the following approaches:

    1. ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator>`
    2. ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator>`
    3. [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`
    4. ``renyi``: :func:`Renyi transfer entropy estimator. <infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator>`
    5. [``ordinal``, ``symbolic``, ``permutation``]: :func:`Ordinal transfer entropy estimator. <infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator>`
    6. ``tsallis``: :func:`Tsallis transfer entropy estimator. <infomeasure.estimators.transfer_entropy.tsallis.TsallisTEEstimator>`

    Parameters
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    approach : str
        The name of the estimator to use.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from source to destination.
        Not compatible with the ``cond`` parameter / conditional TE.
        Alternatively called ``offset``.
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
    if len(data) != 2:
        raise ValueError(
            "Transfer Entropy requires two variables as arguments and if needed,"
            " the conditional data as keyword argument: "
            "`transfer_entropy(source, dest, **kwargs)` or "
            "`transfer_entropy(source, dest, cond=cond_var, **kwargs)`."
        )
    return EstimatorClass(*data, **kwargs).result()


def conditional_transfer_entropy(*data, **kwargs: any):
    """Conditional transfer entropy between two variables given a third variable.

    See :func:`transfer_entropy <transfer_entropy>` for more information.
    """
    if kwargs.get("cond") is None:
        raise ValueError(
            "CTE requires a conditional variable. Pass a 'cond' keyword argument."
        )
    if len(data) != 2:
        raise ValueError(
            "CTE requires two variables as arguments and "
            "the conditional data as keyword argument: "
            "`conditional_transfer_entropy(source, dest, cond=cond_var, **kwargs)`."
        )
    return transfer_entropy(*data, **kwargs)


def estimator(
    *data,  # *data for entropy, *data for mi, *(source, dest) for te
    # all arguments after this are keyword-only
    cond=None,
    measure: str = None,
    approach: str = None,
    step_size: int = 1,
    prop_time: int = 0,
    src_hist_len: int = 1,
    dest_hist_len: int = 1,
    cond_hist_len: int = 1,
    **kwargs: any,
) -> EstimatorType:
    """Get an estimator for a specific measure.

    This function provides a simple interface to get
    an :class:`Estimator <.base.Estimator>` for a specific measure.

    If you are only interested in the global result, use the functional interfaces:

    - :func:`entropy <entropy>`
    - :func:`cross_entropy <cross_entropy>`
    - :func:`mutual_information <mutual_information>`
    - :func:`conditional_mutual_information <conditional_mutual_information>`
    - :func:`transfer_entropy <transfer_entropy>`
    - :func:`conditional_transfer_entropy <conditional_transfer_entropy>`

    Estimators available:

    1. Entropy:
        - ``discrete``: :func:`Discrete entropy estimator. <infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator>`
        - ``kernel``: :func:`Kernel entropy estimator. <infomeasure.estimators.entropy.kernel.KernelEntropyEstimator>`
        - [``metric``, ``kl``]: :func:`Kozachenko-Leonenko entropy estimator. <infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`
        - ``renyi``: :func:`Renyi entropy estimator. <infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator>`
        - [``ordinal``, ``symbolic``, ``permutation``]: :func:`Ordinal / Permutation entropy estimator. <infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator>`
        - ``tsallis``: :func:`Tsallis entropy estimator. <infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator>`

    2. Mutual Information:
        - ``discrete``: :func:`Discrete mutual information estimator. <infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator>`
        - ``kernel``: :func:`Kernel mutual information estimator. <infomeasure.estimators.mutual_information.kernel.KernelMIEstimator>`
        - [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger mutual information estimator. <infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
        - ``renyi``: :func:`Renyi mutual information estimator. <infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator>`
        - [``ordinal``, ``symbolic``, ``permutation``]: :func:`Ordinal mutual information estimator. <infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator>`
        - ``tsallis``: :func:`Tsallis mutual information estimator. <infomeasure.estimators.mutual_information.tsallis.TsallisMIEstimator>`

    3. Transfer Entropy:
        - ``discrete``: :func:`Discrete transfer entropy estimator. <infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator>`
        - ``kernel``: :func:`Kernel transfer entropy estimator. <infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator>`
        - [``metric``, ``ksg``]: :func:`Kraskov-Stoegbauer-Grassberger transfer entropy estimator. <infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`
        - ``renyi``: :func:`Renyi transfer entropy estimator. <infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator>`
        - [``ordinal``, ``symbolic``, ``permutation``]: :func:`Ordinal transfer entropy estimator. <infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator>`
        - ``tsallis``: :func:`Tsallis transfer entropy estimator. <infomeasure.estimators.transfer_entropy.tsallis.TsallisTEEstimator>`

    Parameters
    ----------
    *data :
        The data used to estimate the measure.
        For entropy: a single array-like data. A tuple of data for joint entropy.
        For cross-entropy: two array-like data. Second input RV relative to the first.
        For mutual information: arbitrary number of array-like data.
        For transfer entropy: two array-like data. Source and destination.
    cond : array-like, optional
        Only if the measure is conditional transfer entropy.
    measure : str
        The measure to estimate.
        Options: ``entropy``, ``cross_entropy``, ``mutual_information``,
        ``transfer_entropy``, ``conditional_mutual_information``,
        ``conditional_transfer_entropy``;
        aliases: ``h``, ``hx``, ``mi``, ``te``, ``cmi``, ``cte``.
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
    measure_comp = measure.lower().replace(" ", "_").replace("-", "_")
    if measure_comp in ["entropy", "h", "cross_entropy", "hx"]:
        if len(data) == 0 and measure_comp in ["entropy", "h"]:
            raise ValueError("``data`` is required for entropy estimation.")
        if len(data) < 2 and measure_comp in ["cross_entropy", "hx"]:
            raise ValueError(
                "Cross-entropy requires at least two random variables "
                "passed as positional parameters: `cross_entropy(var1, var2, **kwargs)`"
            )
        if cond is not None:
            raise ValueError(
                "Do not pass ``cond`` for entropy estimation. "
                "Conditional entropy is not explicitly supported."
            )
        if len(data) > 2:
            raise ValueError(
                "One or two data parameters are required for entropy estimation. "
                f"Got {len(data)}. "
                "To signal that you want to compute joint entropy, "
                "pass your data in a tuple, e.g., estimator((data1, data2)). "
                "For cross-entropy, pass two separate RVs, e.g., estimator(p, q)."
            )
        EstimatorClass = _get_estimator(entropy_estimators, approach)
        return EstimatorClass(*data, **kwargs)
    elif measure_comp in [
        "mutual_information",
        "mi",
        "conditional_mutual_information",
        "cmi",
    ]:
        if measure_comp in ["cmi", "conditional_mutual_information"] and cond is None:
            raise ValueError(
                "No conditional data was provided for conditional mutual information"
                "estimation. Pass ``cond`` to specify the conditional data."
            )
        if len(data) == 0:
            raise ValueError("No data was provided for mutual information estimation.")
        if len(data) == 1:
            logger.warning(
                "Only one data array provided for mutual information estimation. "
                "Using normal entropy estimator."
            )
            EstimatorClass = _get_estimator(
                entropy_estimators, mi_entropy_map[approach]
            )
            return EstimatorClass(data[0], **kwargs)
        if cond is not None:
            EstimatorClass = _get_estimator(cmi_estimators, approach)
            return EstimatorClass(*data, cond=cond, **kwargs)
        else:
            EstimatorClass = _get_estimator(mi_estimators, approach)
            return EstimatorClass(*data, **kwargs)
    elif measure_comp in [
        "transfer_entropy",
        "te",
        "conditional_transfer_entropy",
        "cte",
    ]:
        if measure_comp in ["cte", "conditional_transfer_entropy"] and cond is None:
            raise ValueError(
                "No conditional data was provided for conditional transfer entropy "
                "estimation. Pass ``cond`` to specify the conditional data."
            )
        if len(data) != 2:
            raise ValueError(
                "Exactly two data arrays are required for transfer entropy estimation."
            )
        if cond is not None:
            EstimatorClass = _get_estimator(cte_estimators, approach)
            return EstimatorClass(
                *data,
                cond=cond,
                prop_time=prop_time,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                cond_hist_len=cond_hist_len,
                step_size=step_size,
                **kwargs,
            )
        else:
            EstimatorClass = _get_estimator(te_estimators, approach)
            return EstimatorClass(
                *data,
                prop_time=prop_time,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                step_size=step_size,
                **kwargs,
            )
    else:
        raise ValueError(f"Unknown measure: {measure}")
