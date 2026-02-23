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

The infomeasure package provides multiple estimators for discrete mutual information, using the same comprehensive suite of estimators available for entropy estimation. Each estimator has different strengths and is appropriate for different data characteristics and sample sizes.

## Available Discrete MI Estimators

All discrete entropy estimators are also available for mutual information estimation:

- **Basic Estimators**: `discrete` (MLE), `miller_madow`
- **Bias-Corrected**: `grassberger`, `shrink` (James-Stein)
- **Coverage-Based**: `chao_shen`, `chao_wang_jost`
- **Bayesian**: `bayes`, `nsb`, `ansb`
- **Specialized**: `zhang`, `bonachela`

The choice of estimator depends on your data characteristics, sample size, and bias-variance requirements. For detailed guidance, see the {ref}`estimator_selection_guide`.

All estimators support an arbitrary number of random variables that can be passed to
calculate {ref}`mi-discrete-multi-var`.

## Basic Usage

### Discrete (Maximum Likelihood) Estimator

The simplest estimator uses probability mass functions (pmf) estimated by counting occurrences of the time series. This estimator is computationally efficient but can be biased for small samples.


```{code-cell}
import infomeasure as im
data_x = [0, 1, 0, 1, 0, 1, 0, 1]
data_y = [0, 0, 0, 1, 1, 1, 1, 1]
im.mutual_information(data_x, data_y, approach="discrete")
```

```{admonition} Mathematical Example
$X$ has $p_x(0)=p_x(1)=1/2$ and $Y$ has $p_y(0)=3/8$ and $p_y(1)=5/8$.
The joint distribution is given by $p_{xy}((0, 0))=p_{xy}((0, 1))=1/4$, $p_{xy}((1, 0))=1/8$, and $p_{xy}((1, 1))=3/8$.

$$
\begin{aligned}
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
\end{aligned}
$$
```

### Using Different Estimators

You can use any of the available estimators for mutual information:

```{code-cell}
# Compare different estimators
data_x_small = [0, 1, 2, 0, 1, 0]
data_y_small = [1, 0, 2, 1, 0, 1]

estimators = ["discrete", "miller_madow", "grassberger", "shrink", "chao_shen"]
for estimator in estimators:
    mi = im.mutual_information(data_x_small, data_y_small, approach=estimator)
    print(f"{estimator:15}: {mi:.6f} nats")
```

### Bayesian Estimators

```{code-cell}
# Bayesian estimator with different priors
mi_bayes_jeffrey = im.mutual_information(data_x_small, data_y_small, approach="bayes", alpha=0.5)
mi_bayes_laplace = im.mutual_information(data_x_small, data_y_small, approach="bayes", alpha=1.0)
print(f"Bayesian (Jeffrey): {mi_bayes_jeffrey:.6f} nats")
print(f"Bayesian (Laplace): {mi_bayes_laplace:.6f} nats")
```

### Advanced Estimators

```{code-cell}
# NSB estimator (best for correlated data)
mi_nsb = im.mutual_information(data_x_small, data_y_small, approach="nsb")
print(f"NSB: {mi_nsb:.6f} nats")

# Chao-Wang-Jost (advanced bias correction)
mi_cwj = im.mutual_information(data_x_small, data_y_small, approach="chao_wang_jost")
print(f"Chao-Wang-Jost: {mi_cwj:.6f} nats")
```


## Advanced Usage

### Local Mutual Information and Statistical Testing

{ref}`Local Mutual Information` and {ref}`hypothesis testing` require an estimator instance:

```{code-cell}
# Create estimator instance for advanced analysis
est = im.estimator(data_x, data_y, measure="mi", approach="discrete")

# Calculate local mutual information values
local_mi = est.local_vals()
print(f"Local MI values: {local_mi}")

# Perform statistical testing
stat_test = est.statistical_test(n_tests=50, method="permutation_test")
print(f"P-value: {stat_test.p_value:.4f}")
print(f"T-score: {stat_test.t_score:.4f}")
print(f"90% CI: {stat_test.confidence_interval(0.90)}")
print(f"Median of null distribution: {stat_test.percentile(50):.4f}")
```

(mi-discrete-multi-var)=
### Multi-variable Mutual Information

For three or more variables, add them as positional parameters:

```{code-cell}
data_z = [0, 0, 1, 0, 0, 0, 1, 0]
mi_multivar = im.mutual_information(data_x, data_y, data_z, approach="discrete")
print(f"Multi-variable MI: {mi_multivar:.6f}")

# Compare with different estimators
mi_multivar_mm = im.mutual_information(data_x, data_y, data_z, approach="miller_madow")
print(f"Multi-variable MI (Miller-Madow): {mi_multivar_mm:.6f}")
```

### Using Offsets

Introducing the `offset` parameter for time series analysis:

```{code-cell}
mi_offset = im.mutual_information(data_x, data_y, approach="discrete", offset=1)
print(f"MI with offset=1: {mi_offset:.6f}")
```

The `offset` is available when using exactly two RVs and moves the samples relative
to each other.
For example `data_x=[1, 2, 3, 4]` and `data_y=[2, 2, 3, 3]` with `offset=1` will then
match `[1, 2, 3]` and `[2, 3, 3]`.

The estimator is implemented in the {py:class}`DiscreteMIEstimator <infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator>` class,
which is part of the {py:mod}`im.measures.mutual_information <infomeasure.estimators.mutual_information>` module.

```{eval-rst}
.. autoclass:: infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator
    :noindex:
    :undoc-members:
    :show-inheritance:
```
