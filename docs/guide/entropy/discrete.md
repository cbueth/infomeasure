---
file_format: mystnb
kernelspec:
  name: python3
---

(discrete_entropy)=
# Discrete Entropy Estimation
The Shannon discrete {ref}`entropy <entropy_overview>` formula is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\sum_{x \in X} p(x) \log p(x),
$$

where $p(x)$ is the probability mass function (pmf).

To estimate the entropy of a discrete random variable $X$, our implementation uses a plug-in method. Probabilities are estimated by counting occurrences of each configuration in the dataset, and these frequencies are substituted into the formula above. This estimator is simple and computationally efficient.

```{code-cell}
import infomeasure as im

data = [0, 1, 0, 1, 0, 1, 0, 1]
im.entropy(data, approach="discrete", base=2)
```

In this example data, each state of $0$ or $1$ has a probability of $0.5$, resulting in entropy of
$H(X) = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = -\log_2\left(\tfrac{1}{2}\right) = \log_2 2 = 1$ bit.

For this estimator, access to the distribution dictionary is also available.
```{code-cell}
data = [1, 2, 3, 1, 2, 1, 2, 3]
est = im.estimator(data, measure="h", approach="discrete", base=2)
est.result(), est.data[0]
```

As expected, $\sum_{i=1}^n p_i = 1$.
Local values:

```{code-cell}
from numpy import mean
est.local_vals()
```

To verify the identity of the {ref}`local values <local entropy>`, the mean of the local values $\langle h(x) \rangle$ is equal to the global value $H(X)$.

```{code-cell}
est.result() == mean(est.local_vals())
```

The estimator is implemented in the {py:class}`DiscreteEntropyEstimator <infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.estimators.entropy>` module.


(DiscreteEntropyEstimator_impl)=
```{eval-rst}
.. autoclass:: infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
