---
file_format: mystnb
kernelspec:
  name: python3
---

(kullback_leibler_divergence)=
# Kullback–Leibler Divergence **(KLD)**
The Kullback–Leibler divergence, it is the mathematical measure of difference in between two probability distributions.
It is a measure of **relative entropy**.
E.g.: $P$ and $Q$ are probability distributions over a set $\mathcal{X}$:

$$
\begin{align}
D_{KL}(P \parallel Q) &= \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right)\\
&= H(P, Q) - H(P)
\end{align}
$$
One can interpret the K-L divergence as degree of surprise one encounter by falsely assigning the distribution $Q$ for true distribution $P$ in a model. Even though K-L divergence seems to measure some sort of _distance_ (in a sense) between the two probability distributions, it is not a distance metric in the mathematical sense as it lacks some of the properties such as being symmetric and satisfying the triangle inequality.

```{code-cell}
import infomeasure as im

p = [6, 3, 1, 3, 8, 1, 2, 9, 7, 7, 3, 7, 3, 3, 5, 7, 7, 3, 3, 5]
q = [2, 1, 6, 6, 3, 3, 6, 5, 3, 1, 7, 9, 3, 3, 1, 5, 4, 6, 6, 1]
im.kullback_leiber_divergence(p, q, approach='discrete')
```

As the internal implementation is using the entropy combination,
any `approach` from {ref}`entropy_overview` are supported, as seen in {py:func}`entropy() <infomeasure.entropy>`.

```{code-cell}
(im.kld(p, q, approach='kernel', kernel='box', bandwidth=3),
 im.kld(p, q, approach='kernel', kernel='gaussian', bandwidth=2))
```

```{code-cell}
im.kld(p, q, approach='metric')  # or 'kl'
```

```{code-cell}
(im.kld(p, q, approach='ordinal', embedding_dim=2),
 im.kld(p, q, approach='ordinal', embedding_dim=3),
 im.kld(p, q, approach='ordinal', embedding_dim=4))
```

```{code-cell}
(im.kld(p, q, approach='renyi', alpha=0.8),
 im.kld(p, q, approach='tsallis', q=0.9))
```
