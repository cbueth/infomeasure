---
file_format: mystnb
kernelspec:
  name: python3
---

(cross_entropy_overview)=
# Cross-Entropy
Cross-entropy is a measure of the difference between two probability distributions. It quantifies the amount of information needed to encode samples from one distribution using a code optimized for another distribution. This measure is particularly useful in machine learning and information theory.

Let $P$ and $Q$ be two random variables (RVs) with their respective probability distributions $p(x)$ and $q(x)$. The cross-entropy between $P$ and $Q$, denoted as $H_Q(P)$, is defined as:

$$
H_Q(P) = H_\times(P, Q) = -\sum_{x \in X} p(x) \log q(x)
$$

Here, $p(x)$ represents the true distribution of $P$, and $q(x)$ represents the estimated distribution of $Q$. This nomenclature $H_Q(P)$ is adopted from Christopher Olah's blog post [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/#fnref4) to avoid ambiguity with joint entropy.

```{admonition} Understanding Cross-Entropy
:class: hint

Imagine you are at a techno club, and there are two DJs:
- **P**: The DJ playing the music you love (your true preference).
- **Q**: The DJ playing music they think you'll enjoy (their estimation of your preference).

Cross-entropy, $H_Q(P)$, measures how much "energy" you would need to dance to the music played by DJ Q, assuming your true preference comes from DJ P.
If DJ Q's playlist aligns with your taste (P), the cross-entropy is low, and you dance effortlessly.
However, if DJ Q plays tracks far from your preference, the cross-entropy is high, and you struggle to vibe.

This is why cross-entropy is often used in machine learning: it evaluates how well predictions (Q) align with the true data (P).
```


This package provides methods to compute cross-entropy using various approaches (e.g., discrete or continuous). The cross-entropy is accessed by passing two RV data parameters as positional arguments. For example:
- `im.entropy(data_P, data_Q, ...)`
- `im.h(data_P, data_Q, ...)` (shorthand for `im.entropy`)
- `im.cross_entropy(data_P, data_Q, ...)` (dedicated for cross-entropy)
- `im.hx(data_P, data_Q, ...)` (dedicated for cross-entropy)

Internally, these all use the `im.entropy` function. When only one RV is passed, it computes normal entropy; when a tuple of RVs is passed, it computes joint entropy; and when two RVs are passed as separate arguments, it computes cross-entropy.

## Supported Estimators

Cross-entropy is available for the following estimators:

- **Basic Estimators**: `discrete` (MLE), `miller_madow`
- **Bayesian**: `bayes` (with multiple priors)

**Continuous Estimators:**
- **Kernel**: `kernel` (with various kernel types)
- **Metric**: `metric` or `kl` (Kozachenko-Leonenko)
- **Ordinal**: `ordinal` (for time series analysis)

**Generalized Entropies:**
- **Rényi**: `renyi`
- **Tsallis**: `tsallis`

## Cross-Entropy Computation
For the discrete Shannon entropy, the cross-entropy $H_Q(P)$ is computed as follows:

```{code-cell}
import infomeasure as im

data_P = [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
data_Q = [1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 0, 2, 0, 0]

# Cross-entropy between P and Q: H_Q(P) = H_x(P, Q)
h_q_p = im.cross_entropy(data_P, data_Q, approach="discrete")
h_q_p
```

For the cross-entropy $H_Q(P)$, the estimated probability mass function (pmf) $p(x)$ belongs to the RV $P$, and $q(x)$ belongs to the RV $Q$.
This formulation is generalized for other approaches (e.g., continuous).

```{code-cell}
from numpy.random import default_rng
rng = default_rng(921521569)
data_P = rng.normal(0.0, 15, size=200)
data_Q = rng.normal(1.0, 14, size=500)
im.cross_entropy(data_P, data_Q, approach="metric")
```

Examples with `v0.5.0` discrete estimators:

```{code-cell}
# discrete data
data_P = [0, 0, 1, 1, 1, 0, 2, 2, 0, 1, 2, 1, 0, 2, 1]
data_Q = [1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 0, 2, 0, 0]
# Miller-Madow estimator (simple bias correction)
im.cross_entropy(data_P, data_Q, approach="miller_madow")
```

```{code-cell}
# Grassberger estimator (finite sample corrections)
im.cross_entropy(data_P, data_Q, approach="bayes", alpha="min-max")
```

## Relationship to Kullback-Leibler Divergence
The cross-entropy $H_Q(P)$ is used in the calculation of the {ref}`Kullback-Leibler Divergence (KLD) <kullback_leibler_divergence>`,
which is defined as {cite:p}`leonenkoClassRenyiInformation2008`:

$$
\operatorname{KLD}(P \| Q) = H_Q(P) - H(P)
$$

Here, $H(P)$ is the entropy of $P$, and $H_Q(P)$ is the cross-entropy between $P$ and $Q$.
