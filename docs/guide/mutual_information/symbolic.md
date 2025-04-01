---
file_format: mystnb
kernelspec:
  name: python3
---
(symbolic_MI)=
# Symbolic / Permutation MI Estimation
{ref}`mutual_information_overview` (MI) quantifies the information shared between two random variables $X$ and $Y$. For our purpose, let us write the expression of MI in between the two times series  $X_t$ and $Y_t$ as:

$$
I(X_{t}; Y_t) = \sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where,
- $p(x_t,y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

``Symbolic MI estimation`` estimates the required probability density function (_pdf_) based on the ordinal structure. The details on the _pdf_ estimation based on ordinal structure is provided in {ref}`Symbolic / Permutation Entropy Estimation`.

## Implementation
Example usage of symbolic MI estimation...

The estimator is implemented in the {py:class}`SymbolicMIEstimator <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
