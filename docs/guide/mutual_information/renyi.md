---
file_format: mystnb
kernelspec:
  name: python3
---

# Rényi & Tsallis MI Estimation
{ref}`mutual_information_overview` (MI) quantifies the information shared between two random variables $X$ and $Y$. For our purpose, let us write the expression of MI in between the two times series  $X_t$ and $Y_t$ as:

$$
I(X_{t}; Y_t) = \sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where,
- $p(x_t,y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

MI can be further expressed in terms of entropy and joint entropy as follows {cite:p}`khinchin1957mathematical` {cite:p}`cover2012elements`:

$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$
where,
- $H(X)$ is the entropy of $X$,
- $H(Y)$ is the entropy of $Y$,
- $H(X, Y)$ is the **joint entropy** of $X$ and $Y$.

``Rényi MI estimate`` is computed by plugging-in the entropies and the join entropy estimates by using the estimation method explained in the {ref}`Rényi Entropy Estimation <renyi_entropy>`.

``Tsallis MI estimate``  is computed by plugging-in the entropies and the join entropy estimates by using the estimation method explained in the {ref}`Tsallis Entropy Estimation <tsallis_entropy>`.


## Implementation
Example usage of Renyi and Tsallis entropy...

The estimator is implemented in the {py:class}`RenyiMIEstimator <infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
