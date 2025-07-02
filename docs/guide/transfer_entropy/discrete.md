---
file_format: mystnb
kernelspec:
  name: python3
---
(discrete_TE)=
# Discrete TE Estimation

The {ref}`transfer_entropy_overview` from the source process $X(x_n)$ to the target process $Y(y_n)$ in terms of probabilities is written as:

$$
T_{x \rightarrow y}(k, l) = -\sum_{y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)}}
p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})
\log \left( \frac{p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})}
{p(y_{n+1} \mid \mathbf{y}_n^{(l)})} \right)
$$

where
- $y_{n+1}$ is the next state of $Y$ at time $n$,
- $ \mathbf{y}_n^{(l)} = \{y_n, \dots, y_{n-l+1}\} $ is the embedding vector of $Y$ considering the  $ l $ previous states (history length),
- $ \mathbf{x}_n^{(k)} = \{x_n, \dots, x_{n-k+1}\} $ embedding vector of $X$ considering the $ k $ previous states (history length),
- $p(y_{n+1}, \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the joint probability of the next state of $Y$, its history, and the history of $X$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)}, \mathbf{x}_n^{(k)})$ is the conditional probability of next state of $Y$ given the histories of $X$ and $Y$,
- $p(y_{n+1} \mid \mathbf{y}_n^{(l)})$ is the conditional probability of next state of $Y$ given only the history of $Y$.

## Available Discrete TE Estimators

The infomeasure package provides multiple estimators for discrete transfer entropy, using the same comprehensive suite of estimators available for entropy and mutual information estimation. Each estimator has different strengths and is appropriate for different data characteristics and sample sizes.

All discrete entropy estimators are also available for transfer entropy estimation:

- **Basic Estimators**: `discrete` (MLE), `miller_madow`
- **Bias-Corrected**: `grassberger`, `shrink` (James-Stein)
- **Coverage-Based**: `chao_shen`, `chao_wang_jost`
- **Bayesian**: `bayes`, `nsb`, `ansb`
- **Specialized**: `zhang`, `bonachela`

The choice of estimator depends on your data characteristics, sample size, and bias-variance requirements. For detailed guidance, see the {ref}`estimator_selection_guide`.

## Basic Usage

### Discrete (Maximum Likelihood) Estimator

The simplest estimator estimates the required probability mass functions (_pmf_) by counting occurrences of matching configurations in the dataset. This estimator is computationally efficient but can be biased for small samples.

```{code-cell}
import infomeasure as im
import numpy as np
rng = np.random.default_rng(5673267189)

data_x = rng.integers(2, size=1000)
data_y = np.roll(data_x, 1)
data_control = rng.integers(2, size=1000)

# Basic discrete TE estimation
te_coupled = im.transfer_entropy(
    data_x,  # source
    data_y,  # target
    approach='discrete',
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2,
)

te_uncoupled = im.transfer_entropy(
    data_x,  # source
    data_control,  # target
    approach='discrete',
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2,
)

print(f"TE (coupled): {te_coupled:.6f} bits")
print(f"TE (uncoupled): {te_uncoupled:.6f} bits")
```

### Using Different Estimators

You can use any of the available estimators for transfer entropy:

```{code-cell}
# Compare different estimators on smaller data
data_x_small = rng.integers(3, size=100)
data_y_small = np.roll(data_x_small, 1)

estimators = ["discrete", "miller_madow", "grassberger", "zhang", "bonachela"]
for estimator in estimators:
    try:
        te = im.transfer_entropy(
            data_x_small, data_y_small,
            approach=estimator,
            step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
            base=2
        )
        print(f"{estimator:15}: {te:.6f} bits")
    except Exception as e:
        print(f"{estimator:15}: Error - {e}")
```

### Bayesian Estimators

```{code-cell}
# Bayesian estimator with different priors
te_bayes_jeffrey = im.transfer_entropy(
    data_x_small, data_y_small,
    approach="bayes", alpha=0.5,
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2
)

te_bayes_laplace = im.transfer_entropy(
    data_x_small, data_y_small,
    approach="bayes", alpha=1.0,
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2
)

print(f"Bayesian (Jeffrey): {te_bayes_jeffrey:.6f} bits")
print(f"Bayesian (Laplace): {te_bayes_laplace:.6f} bits")
```

### Advanced Estimators

```{code-cell}
# NSB estimator (best for correlated data)
te_nsb = im.transfer_entropy(
    data_x_small, data_y_small,
    approach="nsb",
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2
)

# Chao-Wang-Jost (advanced bias correction)
data_x_cwj = np.concatenate([
    data_x_small,
    [3, 4, 5, 5, 6, 7, 8, 9, 10, 11]  # Add rare values: singletons and doubletons
])
rng.shuffle(data_x_cwj)
data_y_cwj = np.roll(data_x_cwj, 1)
te_cwj = im.transfer_entropy(
    data_x_cwj, data_y_cwj,
    approach="chao_wang_jost",
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2
)

# Chao-Shen (accounts for unobserved species)
te_cs = im.transfer_entropy(
    data_x_small, data_y_small,
    approach="chao_shen",
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2
)

print(f"NSB: {te_nsb:.6f} bits")
print(f"Chao-Wang-Jost: {te_cwj:.6f} bits")
print(f"Chao-Shen: {te_cs:.6f} bits")
```

```{code-cell}
:tags: [remove-cell]
from numpy import set_printoptions
set_printoptions(precision=5, threshold=20)
```

## Advanced Usage

### Local Values and Statistical Testing

For advanced analysis including {ref}`local values <Local Transfer Entropy>`, {ref}`effective_te`, and {ref}`hypothesis testing`, create an estimator instance:

```{code-cell}
# Create data with weaker coupling for statistical testing demonstration
rng_stat = np.random.default_rng(12345)
data_x_stat = rng_stat.integers(4, size=200)
data_y_stat = np.zeros_like(data_x_stat)
for i in range(1, len(data_x_stat)):
    if rng_stat.random() < 0.1:  # Only 10% chance to follow pattern
        data_y_stat[i] = data_x_stat[i-1]
    else:
        data_y_stat[i] = rng_stat.integers(4)

# Create estimator instance for advanced analysis
est = im.estimator(
    data_x_stat,  # source
    data_y_stat,  # target
    measure='te',  # or 'transfer_entropy'
    approach='discrete',
    step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
    base=2,
)

# Calculate local transfer entropy values
local_te = est.local_vals()
print(f"Local TE values (first 10): {local_te[:10]}")

# Calculate effective transfer entropy
effective_te = est.effective_val()
print(f"Effective TE: {effective_te:.6f} bits")

# Perform statistical testing
stat_test = est.statistical_test(n_tests=50, method="permutation_test")
print(f"P-value: {stat_test.p_value:.4f}")
print(f"T-score: {stat_test.t_score:.4f}")
print(f"90% CI: {stat_test.confidence_interval(0.90)}")
print(f"Median of null distribution: {stat_test.percentile(50):.4f}")
```

### Using Different Estimators for Advanced Analysis

```{code-cell}
# Compare effective TE with different estimators
estimators_advanced = ["discrete", "miller_madow", "zhang", "bonachela"]
for estimator in estimators_advanced:
    est_adv = im.estimator(
        data_x_small, data_y_small,
        measure='te', approach=estimator,
        step_size=1, prop_time=0, src_hist_len=1, dest_hist_len=1,
        base=2
    )
    effective_te = est_adv.effective_val()
    print(f"{estimator:15} - Effective TE: {effective_te:.6f} bits")
```

## Implementation Details

The discrete transfer entropy estimators are implemented in the following classes:

- {py:class}`DiscreteTEEstimator <infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator>` - Basic MLE estimator
- {py:class}`MillerMadowTEEstimator <infomeasure.estimators.transfer_entropy.miller_madow.MillerMadowTEEstimator>` - Miller-Madow bias correction

All estimators are part of the {py:mod}`infomeasure.estimators.transfer_entropy <infomeasure.estimators.transfer_entropy>` module and support the same interface for local values, effective values, and statistical testing.
