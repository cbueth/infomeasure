---
file_format: mystnb
kernelspec:
  name: python3
---

(discrete_entropy)=
# Discrete Entropy Estimation
The discrete estimator for a discrete RV $X$ following the probability mass function (pmf) $p(x)$ is straight forward act of plug-in probability estimates in the discrete Entropy formula.

```{admonition} Tsallis Entropy
:class: tip
The Shannon {cite:p}`shannonMathematicalTheoryCommunication1948` discrete entropy formula is as follows:

$$
H(X) = -\sum_{x \in X} p(x) \log_b p(x),
$$
where $x$ is the list of realization of RV $X$ with probabilitiy $p(x)$ and ``b`` is the base of the logarithm.
```

````{sidebar} Units of Information
The unit of entropy is the bit when the base of the logarithm is 2.
When the base is 10, the unit is the nat.
Historically, there have been various other units, such as the ban, the hartley, and the shannon.
{cite:p}`iso/tc12IEC800001320082008`
````

In contrast to other implementations, our entropy estimator accepts a list of observations, not a probability distribution.

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
