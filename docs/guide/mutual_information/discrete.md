---
file_format: mystnb
kernelspec:
  name: python3
---
(discrete_MI)=
# Discrete MI Estimation
Mutual Information (MI) quantifies the information shared between two random variables $X$ and $Y$, for more details refer to section {ref}`Mutual Information`.
Let $X_t$ and $Y_t$ represent two discrete time series dataset then the MI in between the two RVs is written as: 

$$
I(X_{t-u}; Y_t) = \sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \log \frac{p(x_{t-u}, y_t)}{p(x_{t-u}) p(y_t)}
$$
where,
- $p(x_t,y_t)$: The joint probability distribution at time $t$.
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities of $X_t$ and $Y_t$, respectively
- $u$: the time lag between two time series.

MI is computed by plugging-in the all the probabilities terms in the above equation. 
The probabilities are estimated by simply counting the matching configurations available in the datasets.

## Implementation
The estimator is implemented in the {py:class}`DiscreteMIEstimator <infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator
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


