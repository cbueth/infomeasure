---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_MI)=
# Kraskov-Stoegbauer-Grassberger (KSG) MI  Estimation
Mutual Information (MI) quantifies the information shared between two random variables $X$ and $Y$, for more details refer to section {ref}`Mutual Information`.
Let $X_t$ and $Y_t$ represent two continuous time series dataset then the MI in between the two RVs is written as: 

$$
I(X_{t-u}; Y_t) = \sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \log \frac{p(x_{t-u}, y_t)}{p(x_{t-u}) p(y_t)}
$$
where,
- $p(x_t,y_t)$: The joint probability distribution at time $t$.
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities of $X_t$ and $Y_t$, respectively
- $u$: the time lag between two time series.

The KSG method sidesteps the need to explicitly calculate these densities as shown in above formula. Instead, it leverages properties of **k-nearest neighbor distances** in the joint space of the variables.
Consider a set of $(N)$ paired observations $\left(x_i, y_i\right)$. For each observation $(i)$, let $r_i$ be the distance to its k-th nearest neighbor in the joint $(X, Y)$ space. 
The key insight of the KSG method is that $r_i$ can be used to estimate local densities.
In the KSG method, mutual information is estimated as:

$$
I(X; Y) = \psi(k) + \psi(N)- \frac{1}{N} \sum_{i=1}^{N} \left[ \psi(n_x(i)) + \psi(n_y(i)) \right]
$$

where:
- $ \psi $ is the digamma function.
- $ N $ is the number of data points.
- $ k $ is the number of nearest neighbors considered.
- $ n_x(i) $ is the number of data points from $X$ within the $k$-th nearest neighbor distance of point $ x_i $ in $X$.
- $ n_y(i) $ is the number of data points from $Y$ within the $k$-th nearest neighbor distance of point $ y_i $ in $Y$.


The basic idea is to estimate the densities in the $X$ and $Y$ spaces using the distances to the $k-th$ nearest neighbors, and then relate these density estimates to mutual information using the **digamma function**.
The methodology is based on the article "Estimating mutual information" {cite:p}`miKSG2004`. 
Authors have purposed two algorithms to estimate the MI in the article, as they have stated "In general they perform very similarly, as far as CPU
 times, statistical errors, and systematic errors are concerned", hence we have implemented only the first algorithm  in this package.   

## Implementation
The estimator is implemented in the {py:class}`KSGMIEstimator <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
