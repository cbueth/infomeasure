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
\operatorname{KLD}(P \parallel Q) &= \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right)\\
&= H_Q(P) - H(P)
\end{align}
$$
Here, $H_Q(P)$ is the {ref}`Cross-Entropy <cross_entropy_overview>` and $H(P)$ the normal {ref}`entropy_overview`.
One can interpret the K-L divergence as degree of surprise one encounter by falsely assigning the distribution $Q$ for true distribution $P$ in a model. Even though K-L divergence seems to measure some sort of _distance_ (in a sense) between the two probability distributions, it is not a distance metric in the mathematical sense as it lacks some of the properties such as being symmetric and satisfying the triangle inequality.

```{code-cell}
import infomeasure as im

p = [6, 3, 1, 3, 8, 1, 2, 9, 7, 7, 3, 7, 3, 3, 5, 7, 7, 3, 3, 5]
q = [2, 1, 6, 6, 3, 3, 6, 5, 3, 1, 7, 9, 3, 3, 1, 5, 4, 6, 6, 1]
im.kullback_leiber_divergence(p, q, approach='discrete')
```

```{admonition} Understanding Kullback-Leibler Divergence
:class: hint

Imagine you are at a techno club, and there are two DJs:
- **P**: The DJ playing your favorite tracks (your true preference).
- **Q**: The DJ playing tracks they think you’ll enjoy (their estimate of your preference).

Kullback-Leibler Divergence (KLD) quantifies the "extra effort" or "surprise" caused by dancing to DJ Q’s playlist **instead of** your ideal playlist.
It measures how much harder it is to vibe to DJ Q's tracks compared to what you'd naturally enjoy.

This differs from cross-entropy $H_Q(P)$, which includes both:
- The baseline effort to dance to your favorite music (the inherent uncertainty of P), and
- The mismatch between DJ Q's playlist  and your true taste (P).

KLD isolates **just the extra effort** caused by the mismatch, **removing the baseline uncertainty of P**.
In other words, it focuses purely on how poorly DJ Q’s playlist aligns with your ideal taste.
In summary:

- Cross-Entropy $H_Q(P)$: The total energy required to dance to DJ Q’s music.
- $\operatorname{KLD}(P \parallel Q)$: The additional burden of dancing to DJ Q’s playlist compared to your perfect playlist.

This is why KLD is often used in information theory and machine learning to evaluate how well an estimated distribution (Q) represents the true data distribution (P).
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
