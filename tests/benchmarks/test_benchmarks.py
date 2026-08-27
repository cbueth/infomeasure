import numpy as np
import pytest

import infomeasure as im


def _correlated(n, corr, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    x = z
    y = corr * z + np.sqrt(1.0 - corr**2) * w
    return x, y


@pytest.mark.benchmark
def test_entropy_grassberger():
    rng = np.random.default_rng(0)
    data = rng.normal(size=2000)
    im.entropy(data, approach="grassberger")


@pytest.mark.benchmark
def test_entropy_discrete():
    rng = np.random.default_rng(1)
    data = rng.integers(0, 10, size=2000)
    im.entropy(data, approach="discrete")


@pytest.mark.benchmark
def test_entropy_ordinal():
    rng = np.random.default_rng(2)
    data = rng.normal(size=1000)
    im.entropy(data, approach="ordinal", embedding_dim=3)


@pytest.mark.benchmark
def test_mi_ksg():
    x, y = _correlated(2000, 0.6, 3)
    im.mutual_information(x, y, approach="ksg", k=4)


@pytest.mark.benchmark
def test_mi_grassberger():
    x, y = _correlated(1500, 0.6, 4)
    im.mutual_information(x, y, approach="grassberger")


@pytest.mark.benchmark
def test_mi_discrete():
    rng = np.random.default_rng(5)
    x = rng.integers(0, 10, size=2000)
    y = rng.integers(0, 10, size=2000)
    im.mutual_information(x, y, approach="discrete")


@pytest.mark.benchmark
def test_mi_ordinal():
    x, y = _correlated(1500, 0.6, 6)
    im.mutual_information(x, y, approach="ordinal", embedding_dim=3)


@pytest.mark.benchmark
def test_te_ksg():
    x, y = _correlated(2000, 0.6, 7)
    im.transfer_entropy(x, y, approach="ksg", k=4)


@pytest.mark.benchmark
def test_te_discrete():
    rng = np.random.default_rng(8)
    x = rng.integers(0, 10, size=2000)
    y = rng.integers(0, 10, size=2000)
    im.transfer_entropy(x, y, approach="discrete")


@pytest.mark.benchmark
def test_cmi_ksg():
    x, y = _correlated(1500, 0.6, 9)
    rng = np.random.default_rng(10)
    z = rng.normal(size=1500)
    im.conditional_mutual_information(x, y, cond=z, approach="ksg", k=4)


@pytest.mark.benchmark
def test_cte_ksg():
    x, y = _correlated(1500, 0.6, 11)
    rng = np.random.default_rng(12)
    z = rng.normal(size=1500)
    im.conditional_transfer_entropy(x, y, cond=z, approach="ksg", k=4)
