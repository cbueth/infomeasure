---
file_format: mystnb
kernelspec:
  name: python3
---

# Jensen–Shannon Distance **(JSD)**
It is worthwhile to start by defining the `Kullback–Leibler (KL) divergence` (relative entropy), it is the mathematical measure of difference in between two probability distributions, eg: $P$ and $Q$ and is represented as $D_{KL}(P \parallel Q)$:

$$
D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$$
One can interpret the K-L divergence as degree of surprise one encounter by falsely assigning the distribution $Q$ for true distribution $P$ in a model. Even though K-L divergence seems to measure some sort of _distance_ (in a sense) between the two probability distributions, it is not a distance metric  in mathematical sense as it lacks some of the properties such as being symmetric and satisfying the triangle inequality.

Hence, a more improvised form of divergence known as `Jensen–Shannon divergence` was purposed as a method of measuring similarity between the probability distributions {cite:p}`Divergence_measures_Lin`. One can say, the Jensen–Shannon Divergence is the symmetrized version of $K-L$ divergence, written as:

$$
\mathrm{JSD}(P \parallel Q) = \frac{1}{2} D(P \parallel M) + \frac{1}{2} D(Q \parallel M)
$$
where, $M = \frac{1}{2} (P + Q)$ is a mixture distribution of $P$ and $Q$ and $D(P \parallel M)$ is the K-L divergence measure.

One can express the above equation in terms of Shannon {ref}`entropy_overview` ($H(X)$) as follows:

$$
\mathrm{JSD}(P \parallel Q) = H\left( \frac{P + Q}{2} \right) - \frac{1}{2} H(P) - \frac{1}{2} H(Q).
$$

 Let $P_1, P_2, \cdots, P_n$ be $n$ probability distributions with weights $\pi_1, \pi_2, \cdots, \pi_n$, respectively. The generalized Jensen-Shannon divergence can be defined as:

$$
JS_{\pi}(P_1, P_2, \cdots, P_n) = H\left( \sum_{i=1}^{n} \pi_i P_i \right) - \sum_{i=1}^{n} \pi_i H(P_i),
$$

where $\pi = (\pi_1, \pi_2, \cdots, \pi_n)$ and $\sum_{i=1}^{n} \pi_i = 1$.
For the case of two probability distributions ($P_1 = P, P_2 =Q$  $ \pi_1 = \pi_2 = 1/2 $) we get back to the expression we started. Hence, the **Jensen–Shannon divergence** can also be understood as the difference between the entropy of average distributions to average of entropies.

```{admonition} Bound of Jensen–Shannon divergence
:class: tip
Important point to note is that Jensen–Shannon divergence measure is bounded by $log_b (n)$, as:

$$
0 \leq \mathrm{JSD}_{\pi_1, \dots, \pi_n} (P_1, P_2, \dots, P_n) \leq \log_b (n)
$$
```

The `Jensen Shannon Distance` (JSD) was purposed as the square root of Jensen Shannon Divergence, ie. $\left[ D_{JS}(P, Q) \right]^{1/2}$ as it comes to fulfill the triangle inequality property required for the distance metric {cite:p}`JSD_distance_Endres`.


> Since JSD is a measure of the similarity between the probability distributions, the larger the value of this metric indicate less the similarity.

## Estimation:
Both the `Jensen Shannon Divergence` and `Jensen Shannon Distance`  (JSD) are estimated by using the three different estimation techniques. The probability distribution function required to compute the these metrics is estimated by the respective estimation method described in hyper-linked pages.

- Discrete estimation [{ref}`discrete_entropy`]
- Ordinal estimation [{ref}`ordinal_entropy`]
- Kernel estimation [{ref}`kernel_entropy`]

## Usage
