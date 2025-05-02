"""Explicit ordinal / permutation entropy estimator tests."""

from itertools import permutations
from math import factorial

import pytest
from numpy.random import default_rng

from infomeasure.estimators.entropy import OrdinalEntropyEstimator


@pytest.mark.parametrize("data_len", [1, 2, 10, 100, 1000, int(1e5)])
@pytest.mark.parametrize("embedding_dim", [1, 2, 3, 4, 5])
def test_ordinal_entropy(data_len, embedding_dim, default_rng):
    """Test the ordinal entropy estimator."""
    data = default_rng.integers(0, 10, data_len)
    if embedding_dim == 1:
        est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
        assert est.result() == 0
        return
    if embedding_dim > data_len:
        with pytest.raises(ValueError):
            est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
            est.result()
        return
    est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
    assert 0 <= est.global_val() <= est._log_base(data_len)
    est.local_vals()


@pytest.mark.parametrize(
    "data,embedding_dim,base,expected",
    [
        ([0, 1, 0, 1, 0], 2, 2, 1.0),  # 2x(01), 2x(10): log_2(2) = 1
        (
            [0, 1, 2, 3, 4, 5],
            2,
            2,
            0.0,
        ),  # 5x(01), 0x(10): log_2(1) = 0
        ([0, 2, 4, 3, 1], 3, 3, 1.0),  # 1x(012), 1x(021), 1x(210): log_3(3) = 1
        ([0, 2, 4, 3, 1], 3, 2, 1.584962500721156),
        # 1x(012), 1x(021), 1x(210): log_2(3) = 1.584962500721156
        ([0, 1, 2, 0], 3, 2, 1.0),  # 1x(012), 1x(201), 0x...: log_2(2) = 1
        ([0, 1, 0, 1, 2, 0], 2, 2, 0.9709505944546686),
        # 3x(01), 2x(10): 3/5*log_2(5/3) + 2/5*log_2(5/2) = 0.9709505944546686
        (list(range(10)), 3, 2, 0.0),  # 8x(012): log_2(1) = 0
        (list(range(10)), 3, "e", 0.0),  # 8x(012): log_e(1) = 0
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 2, 2, 0.9798687566511528),
        # 7x(01), 5x(10): 7/12*log_2(12/7) + 5/12*log_2(12/5) = 0.9798687566511528
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 2, "e", 0.6791932659915256),
        # 7x(01), 5x(10): 7/12*log_e(12/7) + 5/12*log_e(12/5) = 0.6791932659915256
        ([4, 7, 9, 10, 6, 11, 3], 3, 2, 1.5219280948873621),
        # 2x(012), 2x(201), (102): 2*2/5*log_2(5/2) + 1*1/5*log_2(5/1) = 1.5219280948873621
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 3, 2, 2.481714572986073),
        # 3x(021), 2x(120), 2x(012), (210), 2x(102), (201): 1*3/11*log_2(11/3) + 3*2/11*log_2(11/2) + 2*1/11*log_2(11/1) = 2.481714572986073
        ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 5, 2, 3.169925001442312),
        # ([0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8], 12, 2, 1.0),  # TODO: not compatible yet, as code fails trying to generate all 12! combinations
        (["a", "b", "a", "b", "a"], 2, 2, 1.0),  # 2x(10), 2x(0,1): log_2(2) = 1
        ([0.0, 1.0, 0.0, 1.0, 0.0], 2, 2, 1.0),  # 2x(0,1), 2x(10): log_2(2) = 1
        (
            [0, 7, 2, 3, 45, 7, 1, 8, 4, 5, 2, 7, 8, 5, 8, 0, 7, 1, 3, 51, 6, 7],
            4,
            2,
            3.511085408,
        ),
        ([4, 7, 9, 10, 6, 11, 35, 0, 59, 4, 45, 46], 4, 3, 1.7195867761904635),
    ],
)
def test_ordinal_entropy_explicit(data, embedding_dim, base, expected):
    """Test the ordinal entropy estimator with explicit values."""
    assert OrdinalEntropyEstimator(
        data, embedding_dim=embedding_dim, base=base, stable=True
    ).global_val() == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("embedding_dim", [1, 2, 3, 4, 5])
def test_ordinal_entropy_minimum(embedding_dim, default_rng):
    """Test the ordinal entropy estimator with data resulting in minimum entropy."""
    # only increasing values
    data = list(range(10))
    est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
    assert est.result() == 0
    # only decreasing values
    est = OrdinalEntropyEstimator(data[::-1], embedding_dim=embedding_dim)
    assert est.result() == 0
    # only the same value
    data = [0] * 10
    est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
    assert est.result() == 0
    # only one pattern
    data = default_rng.normal(size=embedding_dim)
    est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
    assert est.result() == 0


def uniform_data(
    embedding_dim, min_length=10, rng=default_rng(), max_len_abort=int(1e5)
):
    """Generate uniform data (each pattern should have the same number of occurrences).

    Generates uniquely distributed data for the permutations of embedding_dim 2 and 3.
    For higher embedding_dims is tries to be as uniform as possible,
    but it does not seem possible with this approach.

    Idea:
    - Generate all permutations of the embedding_dim
    - Initialize counter for each pattern with 0
    - Generate the data:
      - Choose a random permutation to start the data with (increment the counter)
      - Repeat until the data has the minimum length and all patterns occur the same number of times:
        - Choose patterns from the counter with minimum occurrences
        - Get the last `embedding_dim-1` elements of the data
        - Filter the patterns that can be appended to the data
        - Choose a random pattern from the filtered patterns
        - Add a number to the data that corresponds to the pattern
          (would generate a pattern with the last `embedding_dim-1` elements
          + the new number) (can be float)
        - Increment the counter for the pattern
    - Return the data
    """
    if embedding_dim < 0:
        raise ValueError("The embedding_dim must be a non-negative integer.")
    if embedding_dim > 20:
        raise ValueError(
            "The embedding_dim is too large. "
            "The number of permutations would be too large."
        )
    all_patterns = list(permutations(range(embedding_dim)))
    counter = {pattern: 0 for pattern in all_patterns}
    # Choose a random permutation to start the data with
    data = list(rng.uniform(-10, 10, size=embedding_dim))
    data = [float(x) for x in data]
    counter[tuple(sorted(range(embedding_dim), key=data.__getitem__))] += 1

    while (len(data) < min_length or len(set(counter.values())) != 1) and len(
        data
    ) <= max_len_abort:
        # Choose patterns from the counter with minimum occurrences
        min_occurrences = min(counter.values())
        candidate_patterns = [
            pattern
            for pattern, occurrences in counter.items()
            if occurrences == min_occurrences
        ]
        # Get the last `embedding_dim-1` elements of the data
        last_elements = data[-(embedding_dim - 1) :]
        # generate middle values between each pair of last elements
        # + a value above and below the outermost values
        candidate_values = [
            (last_elements[i] + last_elements[i + 1]) / 2
            for i in range(len(last_elements) - 1)
        ]
        diff = max(max(last_elements) - min(last_elements), 1)
        candidate_values.extend(
            [
                min(last_elements) - diff / embedding_dim,
                max(last_elements) + diff / embedding_dim,
            ]
        )
        # while loop: choose a random value from the candidate values
        # if new argsort is in the candidate patterns, break
        while True:
            if len(candidate_values) == 0:
                new_value = None
                break

            new_value = float(rng.choice(candidate_values))
            new_argsort = sorted(
                range(len(last_elements) + 1),
                key=lambda i: (last_elements + [new_value])[i],
            )
            if tuple(new_argsort) in candidate_patterns:
                break
            # remove the new value from the candidate values
            candidate_values.remove(new_value)

        if new_value is None:
            # If no optimal candidate could be used, choose a value that makes the
            # last tried pattern more probable in the next round: how?
            new_value = float(
                rng.uniform(
                    min(last_elements) - diff / 2 - 1, max(last_elements) + diff / 2 + 1
                )
            )
            new_argsort = sorted(
                range(len(last_elements) + 1),
                key=lambda i: (last_elements + [new_value])[i],
            )
        # Add the new value to the data
        data.append(new_value)
        # Increment the counter for the pattern
        counter[tuple(new_argsort)] += 1

    return data


@pytest.mark.parametrize("embedding_dim", [2, 3, 4])
@pytest.mark.parametrize("min_length", [10, 100, 1000])
def test_ordinal_entropy_maximum(embedding_dim, min_length, default_rng):
    """Test the ordinal entropy estimator with data resulting in maximum entropy."""
    max_len_abort = int(1e4)
    data = uniform_data(
        embedding_dim,
        min_length=min_length,
        rng=default_rng,
        max_len_abort=max_len_abort,
    )
    est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim, base=2)
    if len(data) < max_len_abort:
        # exact match
        assert est.global_val() == est._log_base(factorial(embedding_dim))
    else:
        # approximate match, should be close to the maximum
        assert pytest.approx(est.global_val(), abs=0.1) == est._log_base(
            factorial(embedding_dim)
        )
        # but cannot be larger
        assert est.global_val() <= est._log_base(factorial(embedding_dim))
    est.local_vals()


@pytest.mark.parametrize("embedding_dim", [-1, 1.0, "a", 1.5, 2.0])
def test_ordinal_entropy_invalid_embedding_dim(embedding_dim):
    """Test the ordinal entropy estimator with invalid embedding_dim."""
    data = list(range(10))
    with pytest.raises(ValueError):
        OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2], [3, 4]],
        [(1, 2), (3, 4)],
        ([1, 2, 3], [[1, 1], [2, 2], [3, 3]]),
        ([1, 2, 3], [[1], [2], [3]]),
        ([1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        [[1], [2], [3]],
        [[1, 1, 1], [2, 2, 2]],
        ([1, 2], [1, 2], [1, 2], [[1, 1], [2, 2]]),
    ],
)
def test_ordinal_entropy_invalid_data_type(data):
    """Test the ordinal entropy estimator with invalid data type."""
    with pytest.raises(
        TypeError,
        match="The data must be a 1D array or tuple of 1D arrays. Ordinal patterns",
    ):
        est = OrdinalEntropyEstimator(data, embedding_dim=2)
        est.result()


@pytest.mark.parametrize(
    "data",
    [
        [None, None, None, None],  # Not comparable
        [1, 2, None, None],  # Inhomogeneous
    ],
)
@pytest.mark.parametrize("embedding_dim", [2, 3, 4])
def test_ordinal_entropy_type_incomparable(data, embedding_dim):
    """Test the ordinal entropy estimator with incomparable data type."""
    with pytest.raises(
        TypeError,
        match="'<' not supported between instances of",
    ):
        est = OrdinalEntropyEstimator(data, embedding_dim=embedding_dim)
        est.result()
