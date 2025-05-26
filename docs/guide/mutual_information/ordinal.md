---
file_format: mystnb
kernelspec:
  name: python3
---
(ordinal_MI)=
# Ordinal / Symbolic / Permutation MI Estimation
{ref}`mutual_information_overview` quantifies the information shared between two random variables $X$ and $Y$. For our purpose, let us write the expression of MI in between the two times series $X_t$ and $Y_t$ as:

$$
I(X_{t}; Y_t) = \sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where
- $p(x_t, y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

Ordinal MI estimation estimates the required probability density function (_pdf_) based on the ordinal structure. The details on the _pdf_ estimation based on ordinal structure is provided in {ref}`Ordinal / Symbolic / Permutation Entropy Estimation`.


To demonstrate this MI, we generate a multivariate Gaussian distribution with two dimensions.
The data is centered around the origin and has a correlation coefficient of $\rho = 0.7$.
The analytical equation of the other approaches does not hold; as for ordinal entropy, the pmf of the ordinal patterns is analyzed.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(692475)

rho = 0.7
data = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=1000)
x, y = data[:, 0], data[:, 1]

im.mutual_information(x, y, approach="ordinal", embedding_dim=3)
```

```{code-cell}
:tags: [remove-cell]
np.set_printoptions(precision=5, threshold=20)
```

Introducing the `offset`:

```{code-cell}
im.mutual_information(x, y, approach="ordinal", embedding_dim=4, offset=1)
```

For three or more variables, add them as positional parameters.

```{code-cell}
data = rng.multivariate_normal([0, 0, 0], [[1, rho, 0], [rho, 1, 0], [0, 0, 1]], size=1000)
data_x, data_y, data_z = data[:, 0], data[:, 1], data[:, 2]
im.mutual_information(data_x, data_y, data_z, approach="ordinal", embedding_dim=2)
```

{ref}`Local Mutual Information` and {ref}`hypothesis testing` need an estimator instance.

```{code-cell}
est = im.estimator(data_x, data_y, measure="mi", approach="ordinal", embedding_dim=2)
est.local_vals(), est.p_value(n_tests = 50, method="permutation_test"), est.t_score()
```

The estimator is implemented in the {py:class}`OrdinalMIEstimator <infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
