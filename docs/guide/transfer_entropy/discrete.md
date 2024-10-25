---
file_format: mystnb
kernelspec:
  name: python3
---

# Discrete Transfer Entropy (TE) Estimation

Transfer entropy is a measure of the amount of information that one time series provides about another time series.

$$
TE(X \rightarrow Y) = H(Y|Y_{-1}) - H(Y|X, Y_{-1})
$$

where ... {cite:p}`schreiberMeasuringInformationTransfer2000`.

```bibtex
@article{schreiberMeasuringInformationTransfer2000,
  title = {Measuring Information Transfer},
  author = {Schreiber, Thomas},
  journal = {Phys. Rev. Lett.},
  volume = {85},
  issue = {2},
  pages = {461--464},
  numpages = {0},
  year = {2000},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.85.461},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.85.461}
}
```

```{admonition} Granger Causality
:class: tip
Transfer entropy is equivalent to Granger causality for Gaussian variables {cite:p}`barnettGrangerCausalityTransfer2009`.
```

```bibtex
@article{barnettGrangerCausalityTransfer2009,
  title = {Granger Causality and Transfer Entropy Are Equivalent for Gaussian Variables},
  author = {Barnett, Lionel and Barrett, Adam B. and Seth, Anil K.},
  journal = {Phys. Rev. Lett.},
  volume = {103},
  issue = {23},
  pages = {238701},
  numpages = {4},
  year = {2009},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.103.238701},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.103.238701}
}
```

```{code-cell}
import infomeasure as im
source = [0, 1, 0, 1, 0, 1, 0, 1]
target = [0, 0, 1, 1, 0, 0, 1, 1]
im.transfer_entropy(source, target, approach="discrete", base=2)
```


The estimator is implemented in the {py:class}`DiscreteTEEstimator <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>` class,
which is part of the {py:mod}`im.measures.transfer_entropy <infomeasure.measures.transfer_entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
