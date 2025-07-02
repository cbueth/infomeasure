---
file_format: mystnb
kernelspec:
  name: python3
---

# Settings

The package configuration can be done using the {py:mod}`im.Config <infomeasure.utils.config.Config>` module.
This will set the default values for the running kernel.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng()
```

## Permanently changing the logarithmic base

The default logarithmic base for the information measures is $e$, ergo, results are in the natural unit of information.
You can change this by using the {py:func}`im.Config.set_logarithmic_unit() <infomeasure.Config.set_logarithmic_unit>` function
or directly setting the base.

```{code-cell}
# To find out the current logarithmic unit and it's description
im.Config.get_logarithmic_unit(), im.Config.get_logarithmic_unit_description()
```

```{code-cell}
im.Config.set_logarithmic_unit("bits")  #  / "shannons"
# equivalent to
im.Config.set("base", 2)  # int | float

im.Config.set_logarithmic_unit("nats")
# equivalent to
im.Config.set("base", "e")  # special value

im.Config.set_logarithmic_unit("hartleys")  # / "bans" / "dits"
# equivalent to
im.Config.set("base", 10)  # int | float
```

Any calculation after this will use the new base.
Only in the case of restarting the kernel, the base will be reset to the default value.
When using multiple bases it is recommended to directly pass the ``base`` argument to the estimator functions, like so:
```{code-cell}
im.entropy([1, 0, 1, 0], approach="discrete", base='e'), \
  im.entropy([1, 0, 1, 0], approach="discrete", base=2)
```

## Statistical Testing Configuration

The package provides comprehensive statistical testing capabilities through the {func}`~infomeasure.estimators.mixins.StatisticalTestingMixin.statistical_test` method.
This method returns a {class}`~infomeasure.utils.data.StatisticalTestResult` object containing _p_-values, _t_-scores, and additional metadata.

### Statistical Test Methods

Two statistical test methods are available:
- **Permutation test** (default): Uses permuted data to generate null distribution
- **Bootstrap test**: Uses resampled data with repetition

The choice depends on your data characteristics and sample size. You can set the default method globally:

```{code-cell}
im.Config.set("statistical_test_method", "permutation_test")  # or "bootstrap"
im.Config.get("statistical_test_method")
```

### Default Number of Tests

You can configure the default number of statistical tests performed:

```{code-cell}
im.Config.set("statistical_test_n_tests", 200)  # default number of tests
im.Config.get("statistical_test_n_tests")
```

### Using Statistical Tests

The statistical testing functionality provides comprehensive results in a single method call:

```{code-cell}
a = rng.integers(0, 2, size=1000)
est = im.estimator(a, np.roll(a, -1), measure="te", approach="discrete")

# Use configuration set by the Config
result_config = est.statistical_test()
print(f"Default: p-value = {result_config.p_value:.4f}, t-score = {result_config.t_score:.4f}")

# Override global settings
result_permutation = est.statistical_test(n_tests=50, method="permutation_test")
print(f"Permutation: p-value = {result_permutation.p_value:.4f}, t-score = {result_permutation.t_score:.4f}")

# Access additional information
print(f"Method used: {result_config.method}")
print(f"Number of tests: {result_config.n_tests}")
print(f"Observed value: {result_config.observed_value:.4f}")
```

### Statistical Test Results

The {class}`~infomeasure.utils.data.StatisticalTestResult` object provides rich information:

```{code-cell}
# Calculate confidence intervals
ci_95 = result_config.confidence_interval(0.95)
print(f"95% Confidence Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

# Access percentiles of the test distribution
median = result_config.percentile(50)
print(f"Median of null distribution: {median:.4f}")

# Access all test values for custom analysis
test_values = result_config.test_values
print(f"Number of test values: {len(test_values)}")
```

## Logging

The package uses the Python {py:mod}`logging` library for logging purposes.
By default, a logger called "infomeasure" is set up with
a {py:class}`NullHandler <logging.NullHandler>`
to avoid any unintended logging output.
You can configure the logging settings to suit your needs.


You can change the logging level using
the {py:func}`im.Config.set_log_level() <infomeasure.Config.set_log_level>` function.
This allows you to control the verbosity of the logging output.
Debug logs will show the progress of the computation.

```{code-cell}
im.Config.set_log_level("DEBUG")  # / "INFO" / "WARNING" / "ERROR" / "CRITICAL"
```

The logging level can be set to one of the standard logging levels provided by
the {py:mod}`logging` module.
This allows you to control the verbosity of the logging output and filter
out less important messages.
If you want to further customize the logging behaviour,
you can access the logger directly and configure it as needed.

```{code-cell}
import logging
logger = logging.getLogger("infomeasure")
logger.getEffectiveLevel()
```
