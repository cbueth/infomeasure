(reference_guide)=
# Reference Guide

On these pages you can find documentation for `infomeasure`.
The package implements these information-theoretic measures—such as {ref}`entropy_overview`,
{ref}`mutual_information_overview`, {ref}`cond_mi_overview`, {ref}`transfer_entropy_overview`,
{ref}`cond_te_overview`, {ref}`Cross-Entropy <cross_entropy_overview>`,
and composite measures like {ref}`jensen_shannon_divergence`—for
both discrete and continuous-valued data.
A variety of estimation techniques are employed to calculate these measures,
including methods such as the Kernel method, Kraskov-Stögbauer-Grassberger algorithm,
among others.
For detailed information on how to use these measures programmatically or
access their underlying implementations, please refer to the {ref}`API Reference`.

```{eval-rst}
.. toctree::
   :maxdepth: 2

   introduction
   estimator_usage
   entropy/index
   cond_entropy/index
   cross_entropy/index
   mutual_information/index
   cond_mi/index
   transfer_entropy/index
   cond_te/index
   KLD
   JSD
```
