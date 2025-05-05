---
file_format: mystnb
kernelspec:
  name: python3
---

# Estimator Usage
This page provides a brief overview of the intended use of the `infomeasure` package.
There are three ways to use the package:

1. Using the utility functions provided in the package: {py:func}`im.entropy <infomeasure.entropy>`, {py:func}`im.mutual_information <infomeasure.mutual_information>`, {py:func}`im.transfer_entropy <infomeasure.transfer_entropy>`, and the conditional counterparts. For a full list, find the exposed {ref}`functions` in the API Reference.

2. Using the {py:class}`Estimator <infomeasure.estimators.base.Estimator>` classes through the quick access: {py:func}`im.estimator() <infomeasure.estimator>`.

3. Directly importing the {ref}`Estimator <estimators>` classes and using them.

Each estimator is described in detail in the following sections,
e.g. {ref}`Entropy <entropy_overview>`, {ref}`Mutual Information <mutual_information_overview>`, and {ref}`Transfer Entropy <transfer_entropy_overview>`.

Before we start, let's import the necessary packages.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng()
```

```{code-cell}
:tags: [remove-cell]
np.set_printoptions(precision=5, threshold=20)
```

## 1. Utility functions

The {ref}`utility functions <functions>` are the most straightforward way to calculate the information measures.
They are designed to be easy to use and provide a quick way to calculate the information measures.

### Entropy

For example, to calculate the {py:func}`entropy() <infomeasure.entropy>` $H(X)$ of a dataset, you can use the following code:

```{code-cell}
x = rng.integers(0, 2, size=1000)  # binary, uniform data
im.entropy(x, approach="discrete")
```

The available approaches can either be found in the documentation of {py:func}`entropy() <infomeasure.entropy>`,
or on the approach pages as chapters of the {ref}`entropy_overview` section.

### Joint Entropy

Calculating joint entropy $H(X_1, X_2, \ldots, X_n)$ is as simple as calling the same entropy function,
but passing a {py:class}`tuple` of random variables as the first argument.

```{code-cell}
y = rng.choice(["a", "b", "c"], size=1000)  # e.g., using strings as symbols
z = rng.choice([True, False], size=1000)  # e.g., using boolean values as symbols
im.entropy((x, y, z), approach="discrete")
```

With these two functions, you can use the chain rule $H(X|Y) = H(X, Y) - H(Y)$ to combine them to calculate the conditional entropy $H(X|Y)$.

### Cross-Entropy

For two RVs $P$ and $Q$,
you can calculate the cross-entropy $H_Q(P)$ as follows:

```{code-cell}
import infomeasure as im

data_P = [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
data_Q = [1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 0, 0, 2, 0, 0]

# Cross-entropy between P and Q: H_Q(P) = H_x(P, Q)
h_q_p = im.cross_entropy(data_P, data_Q, approach="discrete")
h_q_p
```

This formulation is generalized for other approaches (e.g., continuous).

```{code-cell}
from numpy.random import default_rng
rng = default_rng(921521569)
data_P = rng.normal(0.0, 15, size=200)
data_Q = rng.normal(1.0, 14, size=500)
im.cross_entropy(data_P, data_Q, approach="metric")
```

{py:func}`im.cross_entropy() <infomeasure.cross_entropy>` and {py:func}`im.hx() <infomeasure.hx>` are
convenience functions around {py:func}`im.entropy() <infomeasure.entropy>`,
so the initial entropy function can also always be used.

```{code-cell}
im.entropy(data_P, data_Q, approach="metric")
```

### Mutual Information

For {py:func}`mutual information() <infomeasure.mutual_information>` $I(X; Y)$ between two variables $X$ and $Y$, you can use the following code:

```{code-cell}
x = rng.normal(0, 1, 1000)  # e.g., continuous data, gaussian distribution
y = rng.normal(0, 1, 1000)
im.mutual_information(x, y, approach="kernel", bandwidth=0.2, kernel="box")
```

To move the two random variables relative to each other, introduce the keyword `offset`.
Both input variables then are shifted by the given number against the origin,
in opposite directions.
This is useful to investigate temporal relationships between two variables.

An arbitrary number of variables can be passed to calculate the mutual information $I(X_1; \ldots; X_n)$ between them.
This has been called interaction information, among other names.
Each variable needs to be passed as a {term}`var-positional parameter <python:parameter>` and all other variables need to be passed as {term}`keyword-only parameters <python:parameter>`, just like so:

```{code-cell}
z = rng.normal(0, 1, 1000)
w = rng.normal(0, 1, (1000, 2))  # "kernel" also supports multi-dimensional data
im.mutual_information(x, y, z, w, approach="kernel", bandwidth=0.2, kernel="gaussian")
```

The available options for the `approach` are listed in the docstring of {py:func}`mutual information() <infomeasure.mutual_information>`.
An example for all functionality of each approach can be found in the subsections of {ref}`mutual_information_overview`.

### Conditional Mutual Information

{py:func}`Conditional mutual information <infomeasure.conditional_mutual_information>` $I(X; Y | Z)$ can be calculated using:

```{code-cell}
im.conditional_mutual_information(
    x, y, cond=z, approach="kernel", bandwidth=0.2, kernel="box"
)
```

Here, the condition is a keyword-only parameter,
as it is also possible to pass multiple variables for $I(X_1; \ldots; X_n | Z)$.

```{code-cell}
im.conditional_mutual_information(
    x, y, z, cond=w, approach="kernel", bandwidth=0.2, kernel="box"
)
```

You can also directly use the {py:func}`im.mutual_information() <infomeasure.mutual_information>` function, to calculate the conditional mutual information, passing the `cond` parameter.

### Transfer Entropy

For {py:func}`transfer_entropy() <infomeasure.transfer_entropy>` $T_{X\to Y}$, you can use the following code:

```{code-cell}
im.transfer_entropy(x, y, approach="metric", k = 4,
    step_size = 1, prop_time = 0, src_hist_len = 1, dest_hist_len = 1, noise_level=1e-8
)
```

The first given variable is considered as the source variable $X$, the second as the destination variable $Y$.
Calling `im.te(y, x, ...)` calculates the transfer entropy from variable `y` to `x`.
The package does not have insights of the user-assigned variable names.

Analogously to the `offset` in mutual information calculation,
`prop_time` allows you to specify the time lag between the source and destination variables.
Furthermore, `src_hist_len` and `dest_hist_len` specify the length of the history window for source and destination variables respectively.
`step_size`, often denoted as $\tau$ in the context of transfer entropy,
specifies the time step between consecutive observations in the history window.

As for H and MI, the approaches are documented in {py:func}`transfer_entropy() <infomeasure.transfer_entropy>`,
and also approach by approach in the subsections of {ref}`transfer_entropy_overview`.

### Conditional Transfer Entropy

When calculating {py:func}`conditional transfer entropy <infomeasure.conditional_transfer_entropy>` $T_{X\to Y|Z}$,
the same parameters as in the normal transfer entropy are used,
but with an additional random variable `cond`, which specifies the conditioning variable $Z$,
and `cond_hist_len` specifies the length of the history window for $Z$.

```{code-cell}
im.conditional_transfer_entropy(
    x, y, cond=z, approach="ordinal", embedding_dim=3,
    src_hist_len=2, dest_hist_len=2, cond_hist_len=1
)
```

Again, you can also directly use the {py:func}`im.transfer_entropy() <infomeasure.transfer_entropy>` function, to calculate the conditional transfer entropy, passing the `cond` parameter.

### Composite Measures

Jensen-Shannon Divergence and Kullback-Leiber Divergence are also available as composite measures.
They can be accessed from {py:func}`im.jensen_shannon_divergence() <infomeasure.jensen_shannon_divergence>` and {py:func}`im.kullback_leiber_divergence() <infomeasure.kullback_leiber_divergence>` respectively, and can be called like so:

```{code-cell}
jsd = im.jensen_shannon_divergence(x, y, approach='ordinal', embedding_dim=3)
kl = im.kullback_leiber_divergence(x, y, approach='renyi', alpha=1.1)
jsd, kl
```

For the `approach`, the aforementioned types of estimation techniques are available.
All parameters the approach needs, here `embedding_dim`, are passed as keyword arguments.

### Shorthands

For convenience, there are further shorthand functions, respectively {py:func}`im.h() <infomeasure.h>`, {py:func}`im.hx() <infomeasure.hx>`, {py:func}`im.mi() <infomeasure.mi>`, {py:func}`im.te() <infomeasure.te()>`, {py:func}`im.cmi() <infomeasure.cmi>`, {py:func}`im.cte() <infomeasure.cte>`, {py:func}`im.jsd() <infomeasure.jsd>`, and {py:func}`im.kld() <infomeasure.kld>`.
They are aliases and used in the same way as the before mentioned functions.

```{caution}
In all utility functions, data always needs to be passed as {term}`var-positional parameters <python:parameter>`, except the conditional data.

```python
im.mi(x=a, y=b, ...)                  # wrong
im.mi(a, b, ...)                      # correct
im.te(source=a, dest=b, cond=c, ...)  # wrong
im.te(a, b, cond=c, ...)              # correct
```

## 2. Estimator classes

Estimator classes need to be used to obtain more specific results,
like local values, P-values, and t-scores.
`infomeasure` provides a set of classes that are used under the hood for the utility functions we just discussed.
These classes can be used directly to calculate the information measures, or to access specific results and methods.
With the {py:func}`im.estimator() <infomeasure.estimator>` function, you can create an estimator instance:

```{code-cell}
a = rng.integers(0, 10, size=1000)
b = rng.integers(0, 10, size=1000)
est = im.estimator(
    a.astype(int),       # data: x | x, y, ... | source, dest
    measure="entropy",   # "mutual_information", "transfer_entropy", "h", "mi", "te",
                         # "conditional_mutual_information", "cmi",
                         # "conditional_transfer_entropy", "cte"
    approach="discrete"  # "kernel", "metric", "kl", "ksg", "ordinal", "symbolic",
                         # "permutation", "renyi", "tsallis"
    # additional parameters for each approach, e.g. `cond = ...` to conditionalize
)
est.result(), est.local_vals()
```

The {py:func}`im.estimator() <infomeasure.estimator>` function uses the same parameters as the utility functions,
only an additional `measure` needs to specify the type of information to estimate.

### Global value

To access the global value, as returned by the utility functions, we can use the {py:func}`global_val() <infomeasure.estimators.base.Estimator.global_val>` method.
{py:func}`result() <infomeasure.estimators.base.Estimator.result>` is an alias to return the same global value.
Once calculated, as above, asking for the same value again will not recalculate it.

```{code-cell}
est.global_val(), est.result()
```

### Local values

To return local values—{ref}`Local Entropy`, {ref}`Local Mutual Information`, {ref}`Local Conditional MI`, {ref}`Local Transfer Entropy`, or {ref}`Local Conditional TE`—use the
{py:func}`local_vals() <infomeasure.estimators.base.Estimator.local_vals>`
method.
```{code-cell}
est.local_vals()
```

### Hypothesis testing

To perform hypothesis testing on the global value of an estimator,
use the {py:func}`p_value() <infomeasure.estimators.base.PValueMixin.p_value>` and
{py:func}`t_score() <infomeasure.estimators.base.PValueMixin.t_score>` methods.
Both mutual information and transfer entropy estimators support hypothesis testing.

```{code-cell}
est = im.estimator(a, b, measure="mutual_information",
                   approach="kernel", bandwidth=0.2, kernel="box")
(est.result(), est.p_value(n_tests = 50, method="permutation_test"),
est.t_score(n_tests = 50, method="permutation_test"))
```

Two methods for resampling are available for hypothesis testing:

* **Permutation test**: This method shuffles the first random variable.
* **Bootstrap**: This method resamples the first random variable with replacement.

Resampling one of the two random variables is removing the relationships between the variables,
and thus used as null hypothesis.

```{code-cell}
est.p_value(method="bootstrap", n_tests=100), est.t_score()
```

When calling both p-value and t-score sequentially,
not passing any parameters in the second call will use the parameters used in the previous call,
as seen in the previous code cell.

### Effective value

With {py:func}`effective_val() <infomeasure.estimators.base.EffectiveValueMixin.effective_val>`
the {ref}`Effective Transfer Entropy <Effective Transfer Entropy (eTE)>` $\operatorname{eTE}$ can be calculated:

```{code-cell}
est = im.estimator(a, b, measure="transfer_entropy", approach="metric",
                   k = 4, step_size = 1, offset = 0,
                   src_hist_len = 1, dest_hist_len = 1, noise_level=1e-8)
est.effective_val()
```

### Available approaches

The {ref}`following table <estimator-functions>` shows the available information measures and estimators, and which methods are available for each estimator.

:::{list-table} Estimator functions
:name: estimator-functions
:widths: 2 1 1 1 1
:header-rows: 1
:stub-columns: 1

*   - Estimator
    - {py:func}`result() <infomeasure.estimators.base.Estimator.result>` {py:func}`global_val() <infomeasure.estimators.base.Estimator.global_val>`
    - {py:func}`local_vals() <infomeasure.estimators.base.Estimator.local_vals>`
    - {py:func}`p_value() <infomeasure.estimators.base.PValueMixin.p_value>` {py:func}`t_score() <infomeasure.estimators.base.PValueMixin.t_score>`
    - {py:func}`effective_val() <infomeasure.estimators.base.EffectiveValueMixin.effective_val>`
*   - {ref}`Entropy <entropy_overview>` & {ref}`Joint Entropy`
    -
    -
    -
    -
*   - {py:class}`Discrete <infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator>`
    - X
    - X
    -
    -
*   - {py:class}`Kernel <infomeasure.estimators.entropy.kernel.KernelEntropyEstimator>`
    - X
    - X
    -
    -
*   - {py:class}`KL <infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`
    - X
    - X
    -
    -
*   - {py:class}`Ordinal <infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator>`
    - X
    - X
    -
    -
*   - {py:class}`Rényi <infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator>`
    - X
    -
    -
    -
*   - {py:class}`Tsallis <infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator>`
    - X
    -
    -
    -
*   - {ref}`Mutual Information <mutual_information_overview>` & {ref}`CMI <Conditional MI>`
    -
    -
    -
    -
*   - {py:class}`Discrete <infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator>`
    - X
    - X
    - X
    -
*   - {py:class}`Kernel <infomeasure.estimators.mutual_information.kernel.KernelMIEstimator>`
    - X
    - X
    - X
    -
*   - {py:class}`KSG <infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
    - X
    - X
    - X
    -
*   - {py:class}`Ordinal <infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator>`
    - X
    - X
    - X
    -
*   - {py:class}`Rényi <infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator>`
    - X
    -
    - X
    -
*   - {py:class}`Tsallis <infomeasure.estimators.mutual_information.tsallis.TsallisMIEstimator>`
    - X
    -
    - X
    -
*   - {ref}`Transfer Entropy <transfer_entropy_overview>` & {ref}`CTE <Conditional TE>`
    -
    -
    -
    -
*   - {py:class}`Discrete <infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator>`
    - X
    - X
    - X
    - X
*   - {py:class}`Kernel <infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator>`
    - X
    - X
    - X
    - X
*   - {py:class}`KSG <infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`
    - X
    - X
    - X
    - X
*   - {py:class}`Ordinal <infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator>`
    - X
    - X
    - X
    - X
*   - {py:class}`Rényi <infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator>`
    - X
    -
    - X
    - X
*   - {py:class}`Tsallis <infomeasure.estimators.transfer_entropy.tsallis.TsallisTEEstimator>`
    - X
    -
    - X
    - X
:::

The methods from the table do the following:

- {py:func}`result() <infomeasure.estimators.base.Estimator.result>` & {py:func}`global_val() <infomeasure.estimators.base.Estimator.global_val>`: Returns the global value of the information measure.
- {py:func}`local_vals() <infomeasure.estimators.base.Estimator.local_vals>`: Returns the local values of the information measure.
- {py:func}`p_value() <infomeasure.estimators.base.PValueMixin.p_value>`: Returns the p-value of the information measure.
- {py:func}`t_score() <infomeasure.estimators.base.PValueMixin.t_score>`: Returns the t-score of the information measure.
- {py:func}`effective_val() <infomeasure.estimators.base.EffectiveValueMixin.effective_val>`: Returns the effective transfer entropy.
- {py:func}`distribution() <infomeasure.estimators.base.DistributionMixin.distribution>`: Returns dictionary of the unique values and their frequencies (just available for discrete and ordinal entropy estimator).

For {ref}`CMI <Conditional MI>` and {ref}`CTE <Conditional TE>`,
the {ref}`hypothesis testing` methods {py:func}`p_value() <infomeasure.estimators.base.PValueMixin.p_value>` and {py:func}`t_score() <infomeasure.estimators.base.PValueMixin.t_score>` are not available, neither the {py:func}`effective_val() <infomeasure.estimators.base.EffectiveValueMixin.effective_val>` method.
This is because the shuffling is not trivial for more than two inputs.
