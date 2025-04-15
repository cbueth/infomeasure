---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_MI)=
# Kraskov-Stoegbauer-Grassberger (KSG) MI  Estimation
{ref}`mutual_information_overview` quantifies the information shared between two random variables $X$ and $Y$. For our purpose, let us write the expression of MI in between the two times series  $X_t$ and $Y_t$ as:

$$
I(X_{t}; Y_t) = -\sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where
- $p(x_t, y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

The KSG method avoids the need to explicitly calculate these densities instead, it leverages properties of **_k-nearest neighbor distances_**, same as in {ref}`Kozachenko-Leonenko (KL) / Metric / kNN Entropy Estimation`).
However, simply using the K-L entropy estimation for estimating the marginal and joint entropies to further estimate the MI would lead to small error, as the errors made from individual estimates would not cancel out due to difference in the dimensionality.
{cite:t}`miKSG2004`, in the article "Estimating mutual information," use the idea that the K-L entropy estimation is valid for any value of $k$ and that its value doesn't need to be fixed while estimating the marginal entropies.

Given two variables $X_i$, $Y_i$, spanning over their marginal spaces, let us consider the joint space $Z_i=(X_i, Y_i)$.
For each observation $(i)$, one can compute $d_i$ as the distance to its k-th nearest neighbor in the joint $Z_i=(X_i, Y_i)$ space by using the maximum norm method, and hence resulting new distances $d_x$ and $d_y$.
Moving forward, the authors purpose two algorithms, as they have stated, "in general, they perform very similarly, as far as CPU times, statistical errors, and systematic errors are concerned," hence we have implemented only the first algorithm in this package.
For the first algorithm, new distances $d_x$ and $d_y$ are taken as $d_i$, and then the number of points $n_x$ and $n_y$ in marginal spaces are counted.
Finally, the average of the sum of digamma functions for each point in the marginal spaces is computed.
This leads to the mutual information between two variables as follows:


$$
I(X; Y) = \psi(k) + \psi(N)- \frac{1}{N} \sum_{i=1}^{N} \left[ \psi(n_x(i)) + \psi(n_y(i)) \right]
$$

where:
- $ \psi $ is the **digamma function**,
- $ N $ is the number of data points,
- $ k $ is the number of nearest neighbors considered,
- $ n_x(\cdot) $ refers to the number of neighbors which are with in a hypercube that defines the search range around a statevector, the size of the hypercube in each of the marginal spaces is defined based on the distance to the $k-th$ nearest neighbor in the highest dimensional space.

For interaction information, the above formula is extended in the sum, and $\psi(N)$ is multiplied by $(1-m)$, with the number of RVs $m$.

To demonstrate this MI, we generate a multivariate Gaussian distribution with two dimensions.
The data is centered around the origin and has a correlation coefficient of $\rho = 0.5$.
For Gaussian random variables, we know the analytical MI is given by:

$$
I(X; Y) = -\frac{1}{2} \log(1 - \rho^2)
$$

where $\rho$ is the Pearson correlation coefficient between $X$ and $Y$.
We then compare this analytical value with the estimated MI using `infomeasure`.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(692475)

rho = 0.5
data = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=1000)
x, y = data[:, 0], data[:, 1]

(im.mutual_information(x, y, approach="metric"),
 -0.5 * np.log(1 - rho**2))  # analytical value
```

```{code-cell}
:tags: [remove-cell]
np.set_printoptions(precision=5, threshold=20)
```

Introducing the `offset`:

```{code-cell}
im.mutual_information(x, y, approach="metric", offset=1)
```

The MI decreases greatly because the offset unmatched the pairs of the generated data.


For three or more variables, add them as positional parameters.

```{code-cell}
data = rng.multivariate_normal([0, 0, 0], [[1, rho, 0], [rho, 1, 0], [0, 0, 1]], size=1000)
data_x, data_y, data_z = data[:, 0], data[:, 1], data[:, 2]
im.mutual_information(data_x, data_y, data_z, approach="metric")
```

{ref}`Local Mutual Information` and {ref}`hypothesis testing` need an estimator instance.

```{code-cell}
est = im.estimator(data_x, data_y, measure="mi", approach="metric")
est.local_vals(), est.p_value(n_tests = 50, method="permutation_test"), est.t_score()
```

The estimator is implemented in the {py:class}`KSGMIEstimator <infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
