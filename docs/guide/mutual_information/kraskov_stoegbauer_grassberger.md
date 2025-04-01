---
file_format: mystnb
kernelspec:
  name: python3
---
(KSG_MI)=
# Kraskov-Stoegbauer-Grassberger (KSG) MI  Estimation
{ref}`mutual_information_overview` (MI) quantifies the information shared between two random variables $X$ and $Y$. For our purpose, let us write the expression of MI in between the two times series  $X_t$ and $Y_t$ as:

$$
I(X_{t}; Y_t) = \sum_{x_{t}, y_t} p(x_{t}, y_t) \log \frac{p(x_{t}, y_t)}{p(x_{t}) p(y_t)}
$$
where,
- $p(x_t,y_t)$ is the joint probability distribution (probability density function, _pdf_),
- $p(x_t)$ and  $p(y_t)$ are the marginal probabilities (_pdf_) of $X_t$ and $Y_t$.

The KSG method sidesteps the need to explicitly calculate these densities instead, it leverages properties of **_k-nearest neighbor distances_** ( {ref}`Kozachenko-Leonenko (KL) / Metric / kNN Entropy Estimation` does the same). However, simply implementing the K-L entropy estimation for estimating the marginal and joint entropies to further estimate the MI would lead to small error, as the errors made from individual estimates would not cancel out due to difference in the dimensionality. Kraskov et. al in the article "Estimating mutual information" {cite:p}`miKSG2004` uses the idea that the K-L entropy estimation is valid for any value of $k$ and that its value does´t need to be fixed while estimating the marginal entropies.

Given two variables $X_i$, $Y_i$ spanning over their marginal spaces, let us consider the  joint space  $(Z_i=(X_i,Y_i)$. For each observation $(i)$, one can compute the $d_i$ as the distance to its k-th nearest neighbor in the joint $(Z_i=(X_i,Y_i)$ space by using the maximum norm method and hence resulting into the new distances $d_x$ and $d_y$. Moving forward author purposes two algorithms, as they have stated "in general they perform very similarly, as far as CPU times, statistical errors, and systematic errors are concerned", hence we have implemented only the first algorithm  in this package. For first algorithm, new distances $d_x$ and $d_y$ are set to maximum, and then the number of points $n_x$ and $n_y$ in marginal spaces are counted. Finally, it is averaged over all the samples and is used to compute the mutual information as shown below:

$$
I(X; Y) = \psi(k) + \psi(N)- \frac{1}{N} \sum_{i=1}^{N} \left[ \psi(n_x(i)) + \psi(n_y(i)) \right]
$$

where:
- $ \psi $ is the **_digamma function_**,
- $ N $ is the number of data points,
- $ k $ is the number of nearest neighbors considered,
- $ n_x(\cdot) $ refers to the number of neighbors which are with in a hypercube that defines the search range around a statevector, the size of the hypercube in each of the marginal spaces is defined based on the distance to the $k-th$ nearest neighbor in the highest dimensional space.


## Implementation
The estimator is implemented in the {py:class}`KSGMIEstimator <infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.measures.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
