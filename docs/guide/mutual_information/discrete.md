---
file_format: mystnb
kernelspec:
  name: python3
---
(discrete_MI)=
# Discrete MI Estimation
{ref}`mutual_information_overview` (MI) quantifies the information shared between two discrete random variables $X$ and $Y$ and expressed as below:

$$
I(X;Y) = \sum_{x, y} p(x, y) \log \frac{p(x,y)}{p(x) p(y)}
$$
where,
- $p(x,y)$ is the joint probability distribution  for the occurrence of joint state $(x,y)$,
- $p(x)$ and $p(y)$ is the marginal probability distribution (probability mass function, _pmf_) of $X$ and $Y$ respectively.

To estimate the MI between two discrete random variable $X$ and $Y$, our implementation uses a plug-in method to the estimated _pmf_. The _pmf_ is estimated by counting occurrences of matching configuration in the dataset by keeping the record of the frequencies. This estimator is simple and computationally efficient.

## Implementation
Example usage of discrete MI estimator...

The estimator is implemented in the {py:class}`DiscreteMIEstimator <infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```

# Usage
```{code-cell}
import infomeasure as im
data_x = [0, 1, 0, 1, 0, 1, 0, 1]
data_y = [0, 0, 1, 1, 0, 0, 1, 1]
im.mutual_information(data_x, data_y, approach="discrete", base=2)
```
