---
file_format: mystnb
kernelspec:
  name: python3
---

(jensen_shannon_divergence)=
# Jensen–Shannon Divergence **(JSD)**
Building onto the {ref}`kullback_leibler_divergence`, Jensen–Shannon divergence was proposed as a method of measuring similarity between the probability distributions {cite:p}`Divergence_measures_Lin`.
One can say, the Jensen–Shannon Divergence is the symmetric version of K-L divergence, written as:

$$
\mathrm{JSD}(P \parallel Q) = \frac{1}{2} D(P \parallel M) + \frac{1}{2} D(Q \parallel M)
$$
where, $M = \frac{1}{2} (P + Q)$ is a mixture distribution of $P$ and $Q$ and $D(P \parallel M)$ is the K-L divergence measure.

One can express the above equation in terms of {ref}`Shannon Entropy` $H(X)$ as follows:

$$
\mathrm{JSD}(P \parallel Q) = H\left(\frac{P + Q}{2} \right) - \frac{1}{2} H(P) - \frac{1}{2} H(Q).
$$

 Let $P_1, P_2, \cdots, P_n$ be $n$ probability distributions with weights $\pi_1, \pi_2, \cdots, \pi_n$, respectively. The generalized Jensen-Shannon divergence can be defined as:

$$
JS_{\pi}(P_1, P_2, \cdots, P_n) = H\left( \sum_{i=1}^{n} \pi_i P_i \right) - \sum_{i=1}^{n} \pi_i H(P_i),
$$

where $\pi = (\pi_1, \pi_2, \cdots, \pi_n)$ and $\sum_{i=1}^{n} \pi_i = 1$.
In the case of two probability distributions $P_1 = P, P_2 =Q$, $ \pi_1 = \pi_2 = 1/2 $ we get back to the expression we started.
Hence, the **Jensen–Shannon divergence** can also be understood as the difference between the entropy of average distributions to average of entropies.

Since JSD is a measure of the similarity between the probability distributions, the larger the value of this metric indicates, the less the similarity.

```{admonition} Bound of Jensen–Shannon divergence
:class: tip
The Jensen–Shannon divergence measure is bounded by $log_b (n)$, as:

$$
0 \leq \mathrm{JSD}_{\pi_1, \dots, \pi_n} (P_1, P_2, \dots, P_n) \leq \log_b (n)
$$
```

The Jensen Shannon **Distance** was proposed as the square root of Jensen Shannon Divergence, i.e., $\left[ D_{JS}(P, Q) \right]^{1/2}$, as it comes to fulfill the triangle inequality property required to make up a metric space {cite:p}`JSD_distance_Endres`.

JSD is compatible with {ref}`discrete_entropy`, {ref}`ordinal_entropy`, and {ref}`kernel_entropy`.

```{code-cell}
import infomeasure as im

p = [6, 3, 1, 3, 8, 1, 2, 9, 7, 7, 3, 7, 3, 3, 5, 7, 7, 3, 3, 5]
q = [2, 1, 6, 6, 3, 3, 6, 5, 3, 1, 7, 9, 3, 3, 1, 5, 4, 6, 6, 1]
im.jensen_shannon_divergence(p, q, approach='discrete')
```

```{code-cell}
(im.jsd(p, q, approach='kernel', kernel='box', bandwidth=3),
 im.jsd(p, q, approach='kernel', kernel='gaussian', bandwidth=2))
```

```{code-cell}
(im.jsd(p, q, approach='ordinal', embedding_dim=2),
 im.jsd(p, q, approach='ordinal', embedding_dim=3),
 im.jsd(p, q, approach='ordinal', embedding_dim=4))
```

Calculating the generalized Jensen-Shannon Divergence is also possible.

```{code-cell}
s = [3, 6, 6, 8, 5, 3, 6, 7, 3, 9, 7, 7, 1, 3, 2, 3, 4, 9, 3, 7]
t = [6, 3, 3, 9, 6, 6, 3, 9, 5, 9, 4, 4, 5, 8, 9, 8, 3, 3, 6, 7]
im.jsd(p, q, s, t, approach='discrete')
```