---
file_format: mystnb
kernelspec:
  name: python3
---

(discrete_entropy)=
# Discrete Entropy Estimation
The Shannon {cite:p}`shannonMathematicalTheoryCommunication1948` discrete entropy formula is given by:

$$
H(X) = -\sum_{x \in X} p(x) \log_b p(x),
$$

where $x$ denotes the realizations of the random variable $X$ with probability $p(x)$, and $b$ is the base of the logarithm. Further details can be found in the section {ref}`Entropy / Uncertainty`.

To estimate the entropy of a discrete random variable $X$, our implementation uses a plug-in method. Unlike other implementations that require a predefined probability distribution, our entropy estimator directly accepts a list of observations. 
Probabilities are estimated by counting occurrences of each configuration in the dataset, and these frequencies are substituted into the formula.
This estimator is simple and computationally efficient, however it currently does not include bias correction techniques yet. 

## Implementation
Let's compute the Shanon entropy.
```{code-cell}
import infomeasure as im

data = [0, 1, 0, 1, 0, 1, 0, 1]
im.entropy(data, approach="discrete", base=2)
```

In this example data, each state of $0$ or $1$ has a probability of $0.5$, resulting in an entropy of
$H(X) = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = -\log_2\left(\tfrac{1}{2}\right) = \log_2 2 = 1$ bit.


The estimator is implemented in the {py:class}`DiscreteEntropyEstimator <infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator>` class,
which is part of the {py:mod}`im.measures.entropy <infomeasure.measures.entropy>` module.


(DiscreteEntropyEstimator_impl)=
```{eval-rst}
.. autoclass:: infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
