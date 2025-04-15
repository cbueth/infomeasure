---
file_format: mystnb
kernelspec:
  name: python3
---
(kernel_MI)=
# Kernel MI Estimation
{ref}`mutual_information_overview` quantifies the information shared between two random variables $X$ and $Y$.
For our purpose, let us write the expression of MI in between the two times series $X_t$ and $Y_t$ as:

$$
I(X_{t}; Y_t) = -\sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where
- $p(x_t,y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

Kernel MI estimation estimates the required probability density function (pdf) via **kernel density estimation (KDE)**.
KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function $(K)$ {cite:p}`silverman1986density`. For more detail on _pdf_ estimation and available **_kernel functions_** check the {ref}`Kernel Entropy Estimation` section.

```{note}
This package offers two different kernel functions: box kernel and gaussian kernel.
 ```

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

(im.mutual_information(x, y, approach="kernel", kernel="box", bandwidth=0.7),
 -0.5 * np.log(1 - rho**2))  # analytical value
```

```{code-cell}
:tags: [remove-cell]
np.set_printoptions(precision=5, threshold=20)
```

And with the gaussian kernel:

```{code-cell}
im.mutual_information(x, y, approach="kernel", kernel="gaussian", bandwidth=0.7)
```

Introducing the `offset`:

```{code-cell}
im.mutual_information(x, y, approach="kernel", kernel="box", bandwidth=0.7, offset=1)
```

The MI decreases greatly because the offset unmatched the pairs of the generated data.


For three or more variables, add them as positional parameters.

```{code-cell}
data = rng.multivariate_normal([0, 0, 0], [[1, rho, 0], [rho, 1, 0], [0, 0, 1]], size=1000)
data_x, data_y, data_z = data[:, 0], data[:, 1], data[:, 2]
im.mutual_information(data_x, data_y, data_z, approach="kernel", kernel="box", bandwidth=0.7)
```

{ref}`Local Mutual Information` and {ref}`hypothesis testing` need an estimator instance.

```{code-cell}
est = im.estimator(data_x, data_y, measure="mi", approach="kernel",
    kernel="gaussian", bandwidth=0.7)
est.local_vals(), est.p_value(n_tests = 50, method="permutation_test"), est.t_score()
```


The estimator is implemented in the {py:class}`KernelMIEstimator <infomeasure.estimators.mutual_information.kernel.KernelMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.kernel.KernelMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
