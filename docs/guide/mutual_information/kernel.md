---
file_format: mystnb
kernelspec:
  name: python3
---
(kernel_MI)=
# Kernel MI Estimation
{ref}`mutual_information_overview` (MI) quantifies the information shared between two random variables $X$ and $Y$. For our purpose, let us write the expression of MI in between the two times series  $X_t$ and $Y_t$ as: 

$$
I(X_{t}; Y_t) = \sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where,
- $p(x_t,y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

``Kernel MI estimation`` estimates the required probability density function (_pdf_)  via **kernel density estimation (KDE)** which provides the probabilities values to be implemented in above formula. KDE estimates density at a reference point by weighting all samples based on their distance from it, using a kernel function $(K)$ {cite:p}`silverman1986density`. For more detail on _pdf_ estimation and available **_kernel functions_** check the {ref}`Kernel Entropy Estimation` section.

```{note}
- This package allows to implement two different kernel functions: box kernel or step kernel.
 ```

## Implementation
Example usage of Kernel MI estimator...


The estimator is implemented in the {py:class}`KernelMIEstimator <infomeasure.measures.mutual_information.kernel.KernelMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.kernel.KernelMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
