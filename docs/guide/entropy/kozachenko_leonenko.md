---
file_format: mystnb
kernelspec:
  name: python3
---

(entropy_kozachenko_leonenko)=
# Kozachenko-Leonenko / Metric / KNN Entropy Estimation

The Kozachenko-Leonenko (KL) entropy estimator is a non-parametric method for estimating the entropy of a continuous random variable based on the nearest neighbor distances of the data points

[//]: # ({cite:p}``. Maybe https://ieeexplore.ieee.org/document/9614144 ?)

```{code-cell}
import infomeasure as im
im.entropy([1, 2, 3, 4, 5], approach="kl")  # or approach="metric"
```


The estimator is implemented in the {py:class}`KozachenkoLeonenkoEntropyEstimator <infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
