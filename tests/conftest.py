"""Module for test fixtures available for all test files"""

import pytest
from numpy.random import default_rng as rng

from infomeasure import Config
from infomeasure.measures import entropy, mutual_information, transfer_entropy

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
    "SymbolicEntropyEstimator": {
        "functional_str": ["symbolic", "permutation"],
        "needed_kwargs": {"order": 2},
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
    "SymbolicMIEstimator": {
        "functional_str": ["symbolic", "permutation"],
        "needed_kwargs": {"order": 2},
    },
    "TsallisMIEstimator": {
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
    "SymbolicTEEstimator": {
        "functional_str": ["symbolic", "permutation"],
        "needed_kwargs": {"order": 2},
    },
    "TsallisTEEstimator": {
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
    params=entropy.__all__,
)
def entropy_estimator(request):
    """A fixture that yields entropy estimator classes, with specific kwargs for one."""
    return getattr(entropy, request.param), ENTROPY_APPROACHES[request.param][
        "needed_kwargs"
    ]


entropy_approach_kwargs = [
    (
        ENTROPY_APPROACHES[est]["functional_str"][i],
        ENTROPY_APPROACHES[est]["needed_kwargs"],
    )
    for est in entropy.__all__
    for i in range(len(ENTROPY_APPROACHES[est]["functional_str"]))
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
    params=mutual_information.__all__,
)
def mi_estimator(request):
    """A fixture that yields mutual information estimator classes."""
    return getattr(mutual_information, request.param), MI_APPROACHES[request.param][
        "needed_kwargs"
    ]


mi_approach_kwargs = [
    (
        MI_APPROACHES[est]["functional_str"][i],
        MI_APPROACHES[est]["needed_kwargs"],
    )
    for est in mutual_information.__all__
    for i in range(len(MI_APPROACHES[est]["functional_str"]))
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
    params=transfer_entropy.__all__,
)
def te_estimator(request):
    """A fixture that yields transfer entropy estimator classes."""
    return getattr(transfer_entropy, request.param), TE_APPROACHES[request.param][
        "needed_kwargs"
    ]


te_approach_kwargs = [
    (
        TE_APPROACHES[est]["functional_str"][i],
        TE_APPROACHES[est]["needed_kwargs"],
    )
    for est in transfer_entropy.__all__
    for i in range(len(TE_APPROACHES[est]["functional_str"]))
]


@pytest.fixture(
    scope="session",
    params=te_approach_kwargs,
    ids=[tak[0] for tak in te_approach_kwargs],
)
def te_approach(request):
    """A fixture that yields a tuple of (approach_str, needed_kwargs)."""
    return request.param
