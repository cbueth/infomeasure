---
file_format: mystnb
kernelspec:
  name: python3
---
(discrete_MI)=
# Discrete MI Estimation
{ref}`mutual_information_overview` quantifies the information shared between two discrete random variables $X$ and $Y$ and expressed as below:

$$
I(X;Y) = \sum_{x, y} p(x, y) \log \frac{p(x,y)}{p(x) p(y)}
$$
where,
- $p(x,y)$ is the joint probability distribution for the occurrence of joint state $(x,y)$,
- $p(x)$ and $p(y)$ is the marginal probability distribution of $X$ and $Y$ respectively.

The probability mass functions (pmf) are estimated by counting occurrences of
the time series $\{x_t\}_{t=1, \ldots, T}$,
as well as $\{y_t\}_{t=1, \ldots, T}$ and the joint
$\{(x_t, y_t)\}_{t=1, \ldots, T}$.
This estimator is simple and computationally efficient.


```{code-cell}
import infomeasure as im
data_x = [0, 1, 0, 1, 0, 1, 0, 1]
data_y = [0, 0, 0, 1, 1, 1, 1, 1]
im.mutual_information(data_x, data_y, approach="discrete")
```

```{admonition} Example
$X$ has $p_x(0)=p_x(1)=1/2$ and $Y$ has $p_y(0)=3/8$ and $p_y(1)=5/8$.
The joint distribution is given by $p_{xy}((0, 0))=p_{xy}((0, 1))=1/4$, $p_{xy}((1, 0))=1/8$, and $p_{xy}((1, 1))=3/8$.

$$
\begin{align}
I(X;Y) &= p_{xy}((0, 0))\ln\frac{p_{xy}((0, 0))}{p_x(0)p_y(0)}
+ p_{xy}((0, 1))\ln\frac{p_{xy}((0, 1))}{p_x(0)p_y(1)}\\
&\quad+ p_{xy}((1, 0))\ln\frac{p_{xy}((1, 0))}{p_x(1)p_y(0)}
+ p_{xy}((1, 1))\ln\frac{p_{xy}((1, 1))}{p_x(1)p_y(1)}\\
&= \frac{1}{4}\ln\frac{1/4}{1/2\cdot 3/8}
+ \frac{1}{4}\ln\frac{1/4}{1/2\cdot 5/8}
+ \frac{1}{8}\ln\frac{1/8}{1/2\cdot 3/8}
+ \frac{3}{8}\ln\frac{3/8}{1/2\cdot 5/8}\\
&= \frac{1}{4}\ln\frac{16}{12} + \frac{1}{4}\ln\frac{16}{20} + \frac{1}{8}\ln\frac{16}{24} + \frac{3}{8}\ln\frac{48}{40}\\
&= \frac{1}{4} \ln(4/3) + \frac{1}{4} \ln(4/5) + \frac{1}{8} \ln(2/3) + \frac{3}{8} \ln(6/5)\\
&=\frac{3}2 \ln(2) - \frac{5}{8} \ln(5)\\
&\approx 0.033822075568605230000373...
\end{align}
$$
```

Introducing the `offset`:

```{code-cell}
im.mutual_information(data_x, data_y, approach="discrete", offset=1)
```

For three or more variables, add them as positional parameters.

```{code-cell}
data_z = [0, 0, 1, 0, 0, 0, 1, 0]
im.mutual_information(data_x, data_y, data_z, approach="discrete")
```

{ref}`Local Mutual Information` and {ref}`hypothesis testing` need an estimator instance.

```{code-cell}
est = im.estimator(data_x, data_y, measure="mi", approach="discrete")
stat_test = est.statistical_test(n_tests=50, method="permutation_test")
est.local_vals(), stat_test.p_value, stat_test.t_score, stat_test.confidence_interval(90), stat_test.percentile(50)
```


The estimator is implemented in the {py:class}`DiscreteMIEstimator <infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
