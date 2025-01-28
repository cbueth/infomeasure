---
file_format: mystnb
kernelspec:
  name: python3
---

# Rényi & Tsallis MI Estimation
Mutual Information (MI) quantifies the information shared between two random variables $X$ and $Y$, for more details refer to section {ref}`Mutual Information`.
Let $X_t$ and $Y_t$ represent two continuous time series dataset then the MI in between the two RVs is written as: 

$$
I(X_{t-u}; Y_t) = \sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \log \frac{p(x_{t-u}, y_t)}{p(x_{t-u}) p(y_t)}
$$
where,
- $p(x_t,y_t)$: The joint probability distribution at time $t$,
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities of $X_t$ and $Y_t$ respectively,
- $u$: the time lag between two time series.

MI can be further expressed in terms of entropy and joint entropy as follows {cite:p}`khinchin1957mathematical` {cite:p}`cover2012elements`:

$$
I(X; Y) = H(X) + H(Y) - H(X, Y) 
$$
where,
- $H(X)$ is the entropy of $X$,
- $H(Y)$ is the entropy of $Y$,
- $H(X, Y)$ is the **joint entropy** of $X$ and $Y$.

**Rényi MI estimate** is computed by plugging-in the entropy and the join entropy estimates by using the estimation method explained in the {ref}`Rényi Entropy Estimation <renyi_entropy>`.
**Tsallis TE estimate** is computed by plugging-in the entropy and the join entropy estimates by using the estimation method explained in the {ref}`Tsallis Entropy Estimation <tsallis_entropy>`.


# Implementation
The estimator is implemented in the {py:class}`RenyiMIEstimator <infomeasure.measures.mutual_information.renyi.RenyiMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.renyi.RenyiMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
