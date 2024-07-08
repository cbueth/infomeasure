---
file_format: mystnb
kernelspec:
  name: python3
---

# Discrete Estimation Techniques

When talking about information-theoretic measures, discrete estimation techniques are the ones that originally come to mind.
Here, we present the most common measures.
For continuous variables, see the following sections.

## Entropy (H) Estimation

The entropy of a discrete random variable is a measure of the uncertainty of the variable.
It is defined as

$$
H(X) = -\sum_{x \in X} p(x) \log_b p(x),
$$

where $X$ is the set of possible values of the random variable and $p(x)$ is the probability of the value $x$ occurring {cite:p}`shannonMathematicalTheoryCommunication1948`.
``b`` is the base of the logarithm.

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
im.estimators.discrete.entropy(data, base=2)
```

In this example data, each state of $0$ or $1$ has a probability of $0.5$, resulting in an entropy of
$H(X) = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = -\log_2\left(\tfrac{1}{2}\right) = \log_2 2 = 1$ bit.


## Mutual Information (MI) Estimator

For MI, we also consider observations of two discrete random variables.
It was also defined by {cite:t}`shannonMathematicalTheoryCommunication1948`.
We count as follows:

1. Let's incrementally count the occurrences of each pair of states $(i)$ and $(j)$ in the two variables, as well as the occurrences of individual states in each variable.

2. Then, compute the average mutual information (MI) using the formula:

    $$
    \text{MI} = \sum_{i,j} p(i, j) \cdot \log_b \left( \frac{p(i, j)}{p(i) \cdot p(j)} \right)
    $$

    Where $p(i, j)$ is the joint probability of $(i)$ and $(j)$, and $p(i)$ and $p(j)$ are the marginal probabilities of $(i)$ and $(j)$, respectively. The function also computes the standard deviation of the local MI values.

3. When
The time difference $\Delta t$ specifies how far apart in time the two variables are when calculating their mutual information.

    Specifically, for each time step $t$, the value of the first variable at $t - \Delta t$ is paired with the value of the second variable at $t$. This is useful in capturing temporal dependencies and understanding how past states of one variable may influence the current state of another.
    Still, the amount of delay $\Delta t$ is not known a priori and must be determined based on the data and the specific problem at hand.

The MI equation remains the same, but with the joint and marginal probabilities calculated using the time-lagged pairs:

$$
\text{MI} = \sum_{i,j} p(i[t-\Delta t], j[t]) \cdot \log_2 \left( \frac{p(i[t-\Delta t], j[t])}{p(i[t-\Delta t]) \cdot p(j[t])} \right)
$$

Here \( p(i[t-\Delta t], j[t]) \) is the joint probability of observing \( i \) at time \( t - \Delta t \) and \( j \) at time \( t \).


## Transfer Entropy Estimator

The Transfer Entropy from the source to the destination variable is calculated using following formula:
$$
TE = \sum p(s_{t-l}, d_{t}, d_{t-k}) \log_2 \left( \frac{p(d_{t} | s_{t-l}, d_{t-k})}{p(d_{t} | d_{t-k})} \right)
$$

- \(TE\): Transfer Entropy
- \(s_{t-l}\): Source state at time \(t-l\)
- \(d_t\): Destination state at time \(t\)
- \(d_{t-k}\): Past state of the destination at time \(t-k\)
- \(p(s_{t-l}, d_t, d_{t-k})\): Joint probability of \(s_{t-l}\), \(d_t\), and \(d_{t-k}\)
- \(p(d_t | s_{t-l}, d_{t-k})\): Conditional probability of \(d_t\) given \(s_{t-l}\) and \(d_{t-k}\)
- \(p(d_t | d_{t-k})\): Conditional probability of \(d_t\) given \(d_{t-k}\)
