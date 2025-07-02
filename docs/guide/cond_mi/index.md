---
file_format: mystnb
kernelspec:
  name: python3
---

(cond_mi_overview)=
# Conditional MI
{ref}`Mutual Information <mutual_information_overview>` (MI) in between two processes $X$ and $Y$ can also be conditioned on another process, such as $Z$, known as conditional MI. Such conditional MI provides the shared information between $X$ and $Y$,
when considering the knowledge of the conditional variable $Z$ and is written as $I(X;Y \mid Z)$.

$$
\begin{align}
I(X;Y \mid Z) &= -\sum_{x, y, z} p(z)p(x,y\mid z) \log \frac{p(x, y \mid z)}{p(x \mid z)p(y \mid z)}\\
&= -\sum_{x, y, z} p(x,y,z) \log \frac{p(x,y,z)p(z)}{p(x,z)p(y,z)}\\
&= H(X \mid Z) - H(X \mid Y,Z)
\end{align}
$$

This package offers calculation of CMI for all approaches that {ref}`mutual_information_overview` offers.
Furthermore, more than two variables are supported.
In this case, CMI is defined as

$$
\begin{align}
I(X_1; X_2; \ldots; X_n \mid Z)&=
-\sum_{x_1, x_2, \ldots, x_n, z} p(z)p(x_1,x_2,\ldots,x_n \mid z) \log \frac{p(x_1,x_2,\ldots,x_n \mid z)}{\prod p(x_i \mid z)}\\
&=-\sum_{x_1, x_2, \ldots, x_n, z} p(x_1,x_2,\ldots,x_n,z) \log \frac{p(x_1,x_2,\ldots,x_n,z)p(z)}{\prod p(x_i, z)}\\
&= - H(X_1, X_2, \ldots, X_n, Z) - H(Z) + \sum_{i=1}^n H(X_i, Z).
\end{align}
$$

## Local Conditional MI
Similar to {ref}`Local Conditional H`, **local or point-wise conditional MI** can be defined as by Fano {cite:p}`fano1961transmission`:

$$
\begin{align}
i(x; y \mid z) &= -\log_b \frac{p(x \mid y, z)}{p(x \mid z)}\\
&= h(x \mid z) - h(x \mid y, z)
\end{align}
$$

The conditional MI can be calculated as the expected value of its local counterparts {cite:p}`Lizier2014`:

$$
I(X; Y \mid Z) = \langle i(x; y \mid z) \rangle.
$$

```{note}
The conditional MI $I(X;Y \mid Z)$ can be either larger or smaller than its non-conditional counter-part, i.e., $I(X; Y)$.
This leads to the idea of **Synergy** and **redundancy** and can be addressed by _information decomposition_ approach {cite:p}`williamsGeneralizedMeasuresInformation2011`.
CMI is symmetric under the same condition $Z$, $I(X;Y \mid Z) = I(Y;X \mid Z)$.
```

This package also allows the user to calculate the {ref}`Local Values` of CMI.

## CMI Estimation
The CMI expression can be expressed in the form of entropies and joint entropies as follows:

$$
I(X;Y \mid Z) = - H(X,Z,Y) + H(X,Z) + H(Z,Y) - H(Z)
$$

While the package uses this formula internally for the Rényi and Tsallis CMI,
all other approaches each are calculated with dedicated, probabilistic implementations.

### Available Discrete Estimators

Conditional mutual information supports all the same discrete estimators as regular mutual information:

- **Basic Estimators**: `discrete` (MLE), `miller_madow`
- **Bias-Corrected**: `grassberger`, `shrink` (James-Stein)
- **Coverage-Based**: `chao_shen`, `chao_wang_jost`
- **Bayesian**: `bayes`, `nsb`, `ansb`
- **Specialized**: `zhang`, `bonachela`

For detailed guidance on estimator selection, see the {ref}`estimator_selection_guide`.

```{code-cell}
import infomeasure as im

x = [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
y = [1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 0, 2, 0, 0]
z = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
cmi = im.cmi(x, y, cond=z, approach='discrete')
cmi_ksg = im.cmi(x, y, cond=z, approach='ksg')
cmi_kernel = im.cmi(x, y, cond=z, approach='kernel', kernel='box', bandwidth=1.5)
cmi_symbolic = im.cmi(x, y, cond=z, approach='symbolic', embedding_dim=3)
cmi, cmi_ksg, cmi_kernel, cmi_symbolic
```

Examples with new `v0.5.0` discrete estimators:

```{code-cell}
# NSB estimator (best for correlated data)
cmi_nsb = im.cmi(x, y, cond=z, approach='nsb')

# Miller-Madow estimator (simple bias correction)
cmi_mm = im.cmi(x, y, cond=z, approach='miller_madow')

# Shrinkage estimator (good for small independent samples)
cmi_shrink = im.cmi(x, y, cond=z, approach='shrink')

print(f"CMI (NSB): {cmi_nsb:.6f}")
print(f"CMI (Miller-Madow): {cmi_mm:.6f}")
print(f"CMI (Shrinkage): {cmi_shrink:.6f}")
```

With four variables, the CMI is calculated as follows:

```{code-cell}
from numpy.random import default_rng
rng = default_rng(917856)
im.cmi(
    rng.normal(size=1000),
    rng.normal(size=1000),
    rng.normal(size=1000),
    rng.normal(size=1000),
    cond=rng.normal(size=1000),
    approach='metric'
)
```

The {ref}`Local Conditional MI` is calculated as follows:

```{code-cell}
est = im.estimator(
    x, y, cond=z,
    measure='cmi',  # or 'conditional_mutual_information'
    approach='discrete'
)
est.local_vals()
```


```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: KSG method

   ksg_cond_mi
```
