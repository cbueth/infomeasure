---
file_format: mystnb
kernelspec:
  name: python3
---

# Estimator Usage

This notebook provides a brief overview of how to use the entropy estimation functions provided in the `infomeasure` package.
There are multiple functional ways to use this package, for entropy, mutual information, (effective) transfer entropy and hypothesis testing (p-value estimation).

There are three ways of addressing the information measures in the `infomeasure` package:

1. Using the utility functions provided in the package: {py:func}`im.entropy <infomeasure.entropy>`, {py:func}`im.mutual_information <infomeasure.mutual_information>`, {py:func}`im.transfer_entropy <infomeasure.transfer_entropy>`.
2. Using the {py:class}`Estimator <infomeasure.measures.base.Estimator>` classes through the quick access: {py:func}`im.estimator() <infomeasure.estimator>`.
3. Directly importing the {py:class}`Estimator <infomeasure.measures.base.Estimator>` classes and using them.

Before we start, let's import the necessary packages and define some data to work with.

```{code-cell}
import infomeasure as im
from numpy.random import default_rng
rng = default_rng()
a = rng.normal(0, 1, 1000)
b = rng.normal(0, 1, 1000)
```

## Using the utility functions

The utility functions are the most straightforward way to calculate the information measures. They are designed to be easy to use and provide a quick way to calculate the information measures.

For example, to calculate the entropy of a dataset, you can use the following code:

```{code-cell}
im.entropy(a.astype(int), approach="discrete")
```

For mutual information, some functions alsoreturn the global value, local values and standard deviation of the local values. For example:

```{code-cell}
glob, local, std = im.mutual_information(
    a, b, approach="kernel", bandwidth=0.2, kernel="box"
)
glob, local[:10], std
```

For transfer entropy, you can use the following code:

```{code-cell}
glob, local, std = im.transfer_entropy(a, b, approach="metric", k = 4,
    step_size = 1, offset = 0, src_hist_len = 1, dest_hist_len = 1, noise_level=1e-8)
glob, local[:10], std
```

Each estimator is described in detail in the following sections,
e.g. {ref}`Entropy <entropy_overview>`, {ref}`Mutual Information <mutual_information_overview>`, and {ref}`Transfer Entropy <transfer_entropy_overview>`.

For convenience, there are further shorthand functions, respectively {py:func}`im.h() <infomeasure.h>`, {py:func}`im.mi() <infomeasure.mi>`, and {py:func}`im.te() <infomeasure.te()>`.
They are aliases and used in the same way as the before mentioned functions.


## Estimator classes

`infomeasure` provides a set of classes that are used under the hood for the utility functions.
These classes can be used directly to calculate the information measures, or to access specific results and methods.
They keep the results and the configuration of the estimator.
With the {py:func}`im.estimator() <infomeasure.estimator>` function, you can create an estimator object:

```{code-cell}
est = im.estimator(data=a.astype(int), measure="entropy", approach="discrete")
est.results()
```

The {py:func}`results() <infomeasure.measures.base.Estimator.results>` method returns the global value, if available the local values and its standard deviation.
This we can see for mutual information:

```{code-cell}
est = im.estimator(data_x=a, data_y=b, measure="mutual_information",
                   approach="kernel", bandwidth=0.2, kernel="box")
glob, local, std = est.results()
glob, local[:10], std
```

If you are only interested in either the ``glob``, ``local``, or ``std`` value, one can use the corresponding methods separately {py:func}`global_val() <infomeasure.measures.base.Estimator.global_val>`, {py:func}`local_val() <infomeasure.measures.base.Estimator.local_val>`, or {py:func}`std_val() <infomeasure.measures.base.Estimator.std_val>`.

```{code-cell}
glob = est.global_val()
local = est.local_val()
std = est.std_val()
glob, local[:10], std
```

One might also return a p-value for the mutual information with {py:func}`p_value() <infomeasure.measures.base.PValueMixin.p_value>`:

```{code-cell}
est.p_value(method="permutation_test", n_permutations=100)
```

Transfer entropy has an additional method to calculate the effective transfer entropy:

```{code-cell}
est = im.estimator(source=a, dest=b, measure="transfer_entropy", approach="metric",
                   k = 4, step_size = 1, offset = 0,
                   src_hist_len = 1, dest_hist_len = 1, noise_level=1e-8)
est.effective_val()
```

### Available approaches

The {ref}`following table <estimator-functions>` shows the available information measures and estimators, and which methods are available for each estimator.

:::{list-table} Estimator functions
:name: estimator-functions
:widths: 2 1 1 1 1 1
:header-rows: 1
:stub-columns: 1

*   - Estimator
    - {py:func}`calculate() <infomeasure.measures.base.Estimator.calculate>` {py:func}`results() <infomeasure.measures.base.Estimator.results>` {py:func}`global_val() <infomeasure.measures.base.Estimator.global_val>`
    - {py:func}`local_val() <infomeasure.measures.base.Estimator.local_val>`
    - {py:func}`std_val() <infomeasure.measures.base.Estimator.std_val>`
    - {py:func}`p_value() <infomeasure.measures.base.PValueMixin.p_value>`
    - {py:func}`effective_val() <infomeasure.measures.base.EffectiveValueMixin.effective_val>`
*   - {ref}`Entropy <entropy_overview>`
    -
    -
    -
    -
    -
*   - {py:class}`Discrete <infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator>`
    - X
    -
    -
    - X
    -
*   - {py:class}`Kernel <infomeasure.measures.entropy.kernel.KernelEntropyEstimator>`
    - X
    -
    -
    - X
    -
*   - {py:class}`KL <infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator>`
    - X
    -
    -
    - X
    -
*   - {py:class}`Symbolic <infomeasure.measures.entropy.symbolic.SymbolicEntropyEstimator>`
    - X
    -
    -
    - X
    -
*   - {py:class}`Rényi <infomeasure.measures.entropy.renyi.RenyiEntropyEstimator>`
    - X
    -
    -
    - X
    -
*   - {py:class}`Tsallis <infomeasure.measures.entropy.tsallis.TsallisEntropyEstimator>`
    - X
    -
    -
    - X
    -
*   - {ref}`Mutual Information <mutual_information_overview>`
    -
    -
    -
    -
    -
*   - {py:class}`Discrete <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>`
    - X
    -
    -
    - X
    - X
*   - {py:class}`Kernel <infomeasure.measures.mutual_information.kernel.KernelMIEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`KSG <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`Symbolic <infomeasure.measures.mutual_information.symbolic.SymbolicMIEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`Rényi <infomeasure.measures.mutual_information.renyi.RenyiMIEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`Tsallis <infomeasure.measures.mutual_information.tsallis.TsallisMIEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {ref}`Transfer Entropy <transfer_entropy_overview>`
    -
    -
    -
    -
    -
*   - {py:class}`Discrete <infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator>`
    - X
    -
    -
    - X
    - X
*   - {py:class}`Kernel <infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`KSG <infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`Symbolic <infomeasure.measures.transfer_entropy.symbolic.SymbolicTEEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`Rényi <infomeasure.measures.transfer_entropy.renyi.RenyiTEEstimator>`
    - X
    - X
    - X
    - X
    - X
*   - {py:class}`Tsallis <infomeasure.measures.transfer_entropy.tsallis.TsallisTEEstimator>`
    - X
    - X
    - X
    - X
    - X
:::

The methods from the table do the following:

- {py:func}`calculate() <infomeasure.measures.base.Estimator.calculate>`: Calculates the information measure, no return value.
- {py:func}`results() <infomeasure.measures.base.Estimator.results>`: Returns the global value and if available the local values and its standard deviation.
- {py:func}`global_val() <infomeasure.measures.base.Estimator.global_val>`: Returns the global value of the information measure.
- {py:func}`local_val() <infomeasure.measures.base.Estimator.local_val>`: Returns the local values of the information measure.
- {py:func}`std_val() <infomeasure.measures.base.Estimator.std_val>`: Returns the standard deviation of the local values.
- {py:func}`p_value() <infomeasure.measures.base.PValueMixin.p_value>`: Returns the p-value of the information measure.
- {py:func}`effective_val() <infomeasure.measures.base.EffectiveValueMixin.effective_val>`: Returns the effective transfer entropy.

Each method can be directly called on the estimator object, it is not necessary to call the {py:func}`calculate() <infomeasure.measures.base.Estimator.calculate>` method before calling the other methods, as it is done internally.
Calling {py:func}`results() <infomeasure.measures.base.Estimator.results>`, {py:func}`global_val() <infomeasure.measures.base.Estimator.global_val>`, {py:func}`local_val() <infomeasure.measures.base.Estimator.local_val>`, or {py:func}`std_val() <infomeasure.measures.base.Estimator.std_val>` twice will return the same values, as the estimator caches the results.

## Package configuration

The package configuration can be done using the {py:mod}`im.Config <infomeasure.utils.config.Config>` module.

### Permanently changing the logarithmic base

The default logarithmic base for the entropy and mutual information calculations is $2$, ergo the unit is the bit.
You can change this by using the {py:func}`im.Config.set_logarithmic_unit() <infomeasure.Config.set_logarithmic_unit>` function
or directly setting the base.

```python
im.Config.set_logarithmic_unit("hartleys")
# equivalent to
im.Config.set("base", 10)  # int | float

# When using nats, specify the base as "nats"
im.Config.set_logarithmic_unit("nats")
# equivalent to
im.Config.set("base", "e")  # special value
```

Any calculation after this will use the new base.
Only in the case of restarting the kernel, the base will be reset to the default value.
When using multiple bases it is recommended to directly pass the ``base`` argument to the estimator functions, like so:
```{code-cell}
im.entropy([1, 0, 1, 0], approach="discrete", base='e'), \
  im.entropy([1, 0, 1, 0], approach="discrete", base=2)
```
