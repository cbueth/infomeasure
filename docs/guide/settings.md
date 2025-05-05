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
im.Config.set_logarithmic_unit("bits")  #  / "shannons"
# equivalent to
im.Config.set("base", 2)  # special value

im.Config.set_logarithmic_unit("hartleys")  # / "bans" / "dits"
# equivalent to
im.Config.set("base", 10)  # int | float

# To find out the current logarithmic unit and it's description
im.Config.get_logarithmic_unit(), im.Config.get_logarithmic_unit_description()
```

Any calculation after this will use the new base.
Only in the case of restarting the kernel, the base will be reset to the default value.
When using multiple bases it is recommended to directly pass the ``base`` argument to the estimator functions, like so:
```{code-cell}
im.entropy([1, 0, 1, 0], approach="discrete", base='e'), \
  im.entropy([1, 0, 1, 0], approach="discrete", base=2)
```

## Permanently changing the hypothesis testing approach

For p-value calculation there are two methods available.
By default, a permutation test is used, but you can also use a bootstrap test.
The permutation test uses permuted data, while the bootstrap test uses with repetition resampled data.
Depending on the sample size and other data characteristics, one method may be more appropriate than the other.
To permanently change the p-value method to "bootstrap", you can set it in the configuration:

```{code-cell}
im.Config.set("p_value_method", "bootstrap")  # / or "permutation_test"
im.Config.get("p_value_method")
```

When calculating p-values, only the number of tests needs to be passed, the method will be automatically selected based on the current configuration.
If specified in the function call, it will override the global setting.

```{code-cell}
a = rng.integers(0, 2, size=1000)
est = im.estimator(a, np.roll(a, -1), measure="te", approach="discrete")
# Use bootstrap method just set with Config
p_bootstrap = est.p_value(n_tests=50)
t_score_bootstrap = est.t_score(n_tests=50)
# Explicitly set to permutation test
p_permutation_test = est.p_value(n_tests=50, method="permutation_test")  # overrides global setting
t_score_permutation_test = est.t_score(n_tests=50, method="permutation_test")
p_bootstrap, t_score_bootstrap, p_permutation_test, t_score_permutation_test
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
If you want to further customize the logging behavior,
you can access the logger directly and configure it as needed.

```{code-cell}
import logging
logger = logging.getLogger("infomeasure")
logger.getEffectiveLevel()
```
