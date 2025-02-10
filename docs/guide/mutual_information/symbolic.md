---
file_format: mystnb
kernelspec:
  name: python3
---
(symbolic_MI)=
# Symbolic / Permutation MI Estimation
Mutual Information (MI) quantifies the information shared between two random variables $X$ and $Y$, for more details refer to section {ref}`Mutual Information`.
Let $X_t$ and $Y_t$ represent two continuous time series dataset then the MI in between the two RVs is written as: 

$$
I(X_{t-u}; Y_t) = \sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \log \frac{p(x_{t-u}, y_t)}{p(x_{t-u}) p(y_t)}
$$
where,
- $p(x_t,y_t)$: The joint probability distribution at time $t$.
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities of $X_t$ and $Y_t$, respectively
- $u$: the time lag between two time series.

The required probabilities on above expression to compute MI  are estimated by the ordinal structure. 
The details on the probability based on ordinal structure is provided in {ref}`Symbolic / Permutation Entropy Estimation` 


## Implementation
The estimator is implemented in the {py:class}`SymbolicMIEstimator <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
