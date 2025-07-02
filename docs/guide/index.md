(reference_guide)=
# Reference Guide

On these pages you can find documentation for `infomeasure`.
The package implements a comprehensive suite of information-theoretic measures—such as {ref}`entropy_overview`,
{ref}`mutual_information_overview`, {ref}`cond_mi_overview`, {ref}`transfer_entropy_overview`,
{ref}`cond_te_overview`, and {ref}`Cross-Entropy <cross_entropy_overview>`—for
both discrete and continuous-valued data.

The package provides multiple estimation techniques for each measure, including bias-corrected estimators,
Bayesian approaches, coverage-based methods, and specialized techniques for different data characteristics.
For discrete data, estimators range from a simple maximum likelihood to sophisticated methods like
NSB, Miller-Madow, and Chao-Wang-Jost.
For continuous data, methods include kernel density estimation
and the Kraskov-Stögbauer-Grassberger algorithm.

For guidance on selecting the appropriate estimator for your data, see the {ref}`estimator_selection_guide`.
For detailed information on programmatic usage and API details, please refer to the {ref}`API Reference`.

```{eval-rst}
.. toctree::
   :maxdepth: 2

   introduction
   estimator_usage
   estimator_selection
   entropy/index
   cond_entropy/index
   cross_entropy/index
   mutual_information/index
   cond_mi/index
   transfer_entropy/index
   cond_te/index
   statistical_tests
   KLD
   JSD
   settings
```
