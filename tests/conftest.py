"""Module for test fixtures available for all test files"""

from functools import cache

import pytest
from numpy import roll, zeros
from numpy.random import default_rng as rng

from infomeasure import Config
from infomeasure.estimators import entropy, mutual_information, transfer_entropy

# Dictionary for each measure with the needed kwargs for the test
# ``functional_str`` should contain all the strings that can be used to address the
#  estimator in the functional API.
# ``needed_kwargs`` should only contain the kwargs that need to be passed to the
#  estimator. All other kwargs should be tested in each dedicated estimator test file.

ENTROPY_APPROACHES = {
    "DiscreteEntropyEstimator": {
        "functional_str": ["discrete"],
        "needed_kwargs": {},
    },
    "KernelEntropyEstimator": {
        "functional_str": ["kernel"],
        "needed_kwargs": {"bandwidth": 0.3, "kernel": "box"},
    },
    "KozachenkoLeonenkoEntropyEstimator": {
        "functional_str": ["metric", "kl"],
        "needed_kwargs": {},
    },
    "OrdinalEntropyEstimator": {
        "functional_str": ["ordinal", "symbolic", "permutation"],
        "needed_kwargs": {"embedding_dim": 2},
    },
    "RenyiEntropyEstimator": {
        "functional_str": ["renyi"],
        "needed_kwargs": {"alpha": 1.5},
    },
    "TsallisEntropyEstimator": {
        "functional_str": ["tsallis"],
        "needed_kwargs": {"q": 2.0},
    },
}

MI_APPROACHES = {
    "DiscreteMIEstimator": {
        "functional_str": ["discrete"],
        "needed_kwargs": {},
    },
    "KernelMIEstimator": {
        "functional_str": ["kernel"],
        "needed_kwargs": {"bandwidth": 0.3, "kernel": "box"},
    },
    "KSGMIEstimator": {
        "functional_str": ["metric", "ksg"],
        "needed_kwargs": {},
    },
    "RenyiMIEstimator": {
        "functional_str": ["renyi"],
        "needed_kwargs": {"alpha": 1.5},
    },
    "OrdinalMIEstimator": {
        "functional_str": ["ordinal", "symbolic", "permutation"],
        "needed_kwargs": {"embedding_dim": 2},
    },
    "TsallisMIEstimator": {
        "functional_str": ["tsallis"],
        "needed_kwargs": {"q": 2.0},
    },
}

CMI_APPROACHES = {
    "DiscreteCMIEstimator": {
        "functional_str": ["discrete"],
        "needed_kwargs": {},
    },
    "KernelCMIEstimator": {
        "functional_str": ["kernel"],
        "needed_kwargs": {"bandwidth": 0.3, "kernel": "box"},
    },
    "KSGCMIEstimator": {
        "functional_str": ["metric", "ksg"],
        "needed_kwargs": {},
    },
    "RenyiCMIEstimator": {
        "functional_str": ["renyi"],
        "needed_kwargs": {"alpha": 1.5},
    },
    "OrdinalCMIEstimator": {
        "functional_str": ["ordinal", "symbolic", "permutation"],
        "needed_kwargs": {"embedding_dim": 2},
    },
    "TsallisCMIEstimator": {
        "functional_str": ["tsallis"],
        "needed_kwargs": {"q": 2.0},
    },
}

TE_APPROACHES = {
    "DiscreteTEEstimator": {
        "functional_str": ["discrete"],
        "needed_kwargs": {},
    },
    "KernelTEEstimator": {
        "functional_str": ["kernel"],
        "needed_kwargs": {"bandwidth": 0.3, "kernel": "box"},
    },
    "KSGTEEstimator": {
        "functional_str": ["metric", "ksg"],
        "needed_kwargs": {},
    },
    "RenyiTEEstimator": {
        "functional_str": ["renyi"],
        "needed_kwargs": {"alpha": 1.5},
    },
    "OrdinalTEEstimator": {
        "functional_str": ["ordinal", "symbolic", "permutation"],
        "needed_kwargs": {"embedding_dim": 2},
    },
    "TsallisTEEstimator": {
        "functional_str": ["tsallis"],
        "needed_kwargs": {"q": 2.0},
    },
}


CTE_APPROACHES = {
    "DiscreteCTEEstimator": {
        "functional_str": ["discrete"],
        "needed_kwargs": {},
    },
    "KernelCTEEstimator": {
        "functional_str": ["kernel"],
        "needed_kwargs": {"bandwidth": 0.3, "kernel": "box"},
    },
    "KSGCTEEstimator": {
        "functional_str": ["metric", "ksg"],
        "needed_kwargs": {},
    },
    "RenyiCTEEstimator": {
        "functional_str": ["renyi"],
        "needed_kwargs": {"alpha": 1.5},
    },
    "OrdinalCTEEstimator": {
        "functional_str": ["ordinal", "symbolic", "permutation"],
        "needed_kwargs": {"embedding_dim": 2},
    },
    "TsallisCTEEstimator": {
        "functional_str": ["tsallis"],
        "needed_kwargs": {"q": 2.0},
    },
}


@pytest.fixture(
    scope="function",
)
def default_rng():
    """A random number generator."""
    return rng(seed=798)


@pytest.fixture(autouse=True, scope="session")
def activate_debug_logging():
    """Activate debug logging for all tests."""
    Config.set_log_level("DEBUG")


@pytest.fixture(
    scope="session",
    params=ENTROPY_APPROACHES.keys(),
)
def entropy_estimator(request):
    """A fixture that yields entropy estimator classes, with specific kwargs for one."""
    return getattr(entropy, request.param), ENTROPY_APPROACHES[request.param][
        "needed_kwargs"
    ]


entropy_approach_kwargs = [
    (elem["functional_str"][i], elem["needed_kwargs"])
    for elem in ENTROPY_APPROACHES.values()
    for i in range(len(elem["functional_str"]))
]


@pytest.fixture(
    scope="session",
    params=entropy_approach_kwargs,
    ids=[eak[0] for eak in entropy_approach_kwargs],
)
def entropy_approach(request):
    """A fixture that yields a tuple of (approach_str, needed_kwargs)."""
    return request.param


@pytest.fixture(
    scope="session",
    params=MI_APPROACHES.keys(),
)
def mi_estimator(request):
    """A fixture that yields mutual information estimator classes."""
    return getattr(mutual_information, request.param), MI_APPROACHES[request.param][
        "needed_kwargs"
    ]


mi_approach_kwargs = [
    (elem["functional_str"][i], elem["needed_kwargs"])
    for elem in MI_APPROACHES.values()
    for i in range(len(elem["functional_str"]))
]


@pytest.fixture(
    scope="session",
    params=mi_approach_kwargs,
    ids=[mak[0] for mak in mi_approach_kwargs],
)
def mi_approach(request):
    """A fixture that yields a tuple of (approach_str, needed_kwargs)."""
    return request.param


@pytest.fixture(
    scope="session",
    params=CMI_APPROACHES.keys(),
)
def cmi_estimator(request):
    """A fixture that yields conditional mutual information estimator classes."""
    return getattr(mutual_information, request.param), CMI_APPROACHES[request.param][
        "needed_kwargs"
    ]


cmi_approach_kwargs = [
    (elem["functional_str"][i], elem["needed_kwargs"])
    for elem in CMI_APPROACHES.values()
    for i in range(len(elem["functional_str"]))
]


@pytest.fixture(
    scope="session",
    params=cmi_approach_kwargs,
    ids=[mak[0] for mak in cmi_approach_kwargs],
)
def cmi_approach(request):
    """A fixture that yields a tuple of (approach_str, needed_kwargs)."""
    return request.param


@pytest.fixture(
    scope="session",
    params=TE_APPROACHES.keys(),
)
def te_estimator(request):
    """A fixture that yields transfer entropy estimator classes."""
    return getattr(transfer_entropy, request.param), TE_APPROACHES[request.param][
        "needed_kwargs"
    ]


te_approach_kwargs = [
    (elem["functional_str"][i], elem["needed_kwargs"])
    for elem in TE_APPROACHES.values()
    for i in range(len(elem["functional_str"]))
]


@pytest.fixture(
    scope="session",
    params=te_approach_kwargs,
    ids=[tak[0] for tak in te_approach_kwargs],
)
def te_approach(request):
    """A fixture that yields a tuple of (approach_str, needed_kwargs)."""
    return request.param


@pytest.fixture(
    scope="session",
    params=CTE_APPROACHES.keys(),
)
def cte_estimator(request):
    """A fixture that yields conditional transfer entropy estimator classes."""
    return getattr(transfer_entropy, request.param), CTE_APPROACHES[request.param][
        "needed_kwargs"
    ]


cte_approach_kwargs = [
    (elem["functional_str"][i], elem["needed_kwargs"])
    for elem in CTE_APPROACHES.values()
    for i in range(len(elem["functional_str"]))
]


@pytest.fixture(
    scope="session",
    params=cte_approach_kwargs,
    ids=[tak[0] for tak in cte_approach_kwargs],
)
def cte_approach(request):
    """A fixture that yields a tuple of (approach_str, needed_kwargs)."""
    return request.param


@cache
def generate_autoregressive_series(rng_int, alpha, beta, gamma, length=1000, scale=10):
    # Initialize the series with zeros
    X = zeros(length)
    Y = zeros(length)
    generator = rng(rng_int)
    # Generate the series
    for i in range(length - 1):
        eta_X = generator.normal(loc=0, scale=scale)
        eta_Y = generator.normal(loc=0, scale=scale)
        X[i + 1] = alpha * X[i] + eta_X
        Y[i + 1] = beta * Y[i] + gamma * X[i] + eta_Y
    return X, Y


@cache
def generate_autoregressive_series_condition(
    rng_int, alpha: tuple, beta, gamma: tuple, length=1000, scale=10
):
    # Initialize the series with zeros
    X = zeros(length)
    Y = zeros(length)
    Z = zeros(length)
    generator = rng(rng_int)
    # Generate the series
    for i in range(length - 1):
        eta_X = generator.normal(loc=0, scale=scale)
        eta_Y = generator.normal(loc=0, scale=scale)
        eta_Z = generator.normal(loc=0, scale=scale)
        X[i + 1] = alpha[0] * X[i] + eta_X
        Z[i + 1] = alpha[1] * Z[i] + eta_Z
        Y[i + 1] = beta * Y[i] + gamma[0] * X[i] + gamma[1] * Z[i] + eta_Y

    return X, Y, Z


@cache
def discrete_random_variables(rng_int, prop_time=0, low=0, high=4, length=1000):
    """Generate two coupled discrete random variables.

    The first variable is a uniform random variable with values in [low, high-1].
    Variable 2 takes the highest bit of the previous value of Variable 1
    (if we take a 2 bit representation of variable 1)
    as its own lowest bit, then assigns its highest bit at random.

    So, the two should have ~1 bit of mutual information.
    """
    generator = rng(rng_int)
    X = generator.integers(low, high, length)
    Y = [0] * length
    for i in range(1, length):
        Y[i] = (X[i - 1 - prop_time] & 1) + (generator.integers(0, 2) << 1)
    return X, Y


@cache
def discrete_random_variables_shifted(rng_int, shift=1, low=0, high=2, length=1000):
    """
    Generate the source array filled with random integers in [low, high[.
    Create the destination array by circularly shifting the source array by `prop_time` index.

    A third random array is generated, uncoupled from the first two.

    Parameters
    ----------
    rng_int : int
        Random number generator seed.
    shift : int
        Time delay.
    low : int
        low
    high : int
        high
    length : int
        Length of the output arrays.

    Returns
    -------
    X : ndarray
        Source array.
    Y : ndarray
        Destination array.
    C : ndarray
        Control array.
    """
    generator = rng(rng_int)
    X = generator.integers(low, high, length)
    Y = roll(X, shift)  # e.g., roll([0, 1, 2, 3], 1) = [3, 0, 1, 2]
    return X, Y, generator.integers(low, high, length)


@cache
def discrete_random_variables_condition(rng_int, low=0, high=4, length=1000):
    """Generate three coupled discrete random variables.

    The first variable is a uniform random variable with values in [low, high-1].
    Variable 2 takes the highest bit of the previous value of Variable 1
    (if we take a 2 bit representation of variable 1)
    as its own lowest bit, then assigns its highest bit at random.
    Variable 3 takes the lowest bit of the previous value of Variable 1
    and the highest bit of the previous value of Variable 2.
    """
    generator = rng(rng_int)
    X = generator.integers(low, high, length)
    Y = [0] * length
    Z = [0] * length
    for i in range(1, length):
        Y[i] = (X[i - 1] & 1) + (generator.integers(0, 2) << 1)
        Z[i] = (X[i - 1] & 1) + (Y[i - 1] & 2)
    return X, Y, Z
