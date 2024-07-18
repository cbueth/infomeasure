---
file_format: mystnb
kernelspec:
  name: python3
---

# Discrete Mutual Information (MI) Estimation

The mutual information between two discrete random variables is a measure of the amount of information that one variable provides about the other.

$$
MI(X;Y) =
$$

where ... {cite:p}`shannonMathematicalTheoryCommunication1948`.
``b`` is the base of the logarithm.

```{code-cell}
import infomeasure as im
data_x = [0, 1, 0, 1, 0, 1, 0, 1]
data_y = [0, 0, 1, 1, 0, 0, 1, 1]
im.mutual_information(data_x, data_y, approach="discrete", base=2)
```

In comparison to the {ref}`Discrete Entropy Estimation <discrete_entropy>`, ...


The estimator is implemented in the {py:class}`DiscreteMIEstimator <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
