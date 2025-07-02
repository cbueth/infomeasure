---
file_format: mystnb
kernelspec:
  name: python3
---

(discrete_entropy)=
# Discrete Entropy Estimation

The Shannon discrete {ref}`entropy <entropy_overview>` formula is given as {cite:p}`shannonMathematicalTheoryCommunication1948`:

$$
H(X) = -\sum_{x \in X} p(x) \log p(x),
$$

where $p(x)$ is the probability mass function (pmf).

The infomeasure package provides multiple estimators for discrete entropy, each with different strengths and appropriate use cases. The choice of estimator depends on factors such as sample size, presence of correlations, and desired bias-variance trade-offs.

## Available Discrete Entropy Estimators

### Basic Estimators

#### Discrete (Maximum Likelihood)
The simplest estimator uses a plug-in method where probabilities are estimated by counting occurrences of each configuration in the dataset. This estimator is computationally efficient but can be biased for small samples.

$$
\hat{H} = -\sum_{i=1}^{K} \hat{p}_i \log \hat{p}_i
$$

where $\hat{p}_i = \frac{n_i}{N}$ are the empirical probabilities, $n_i$ are the counts for each unique value $i$, $K$ is the number of unique values, and $N$ is the total number of observations.

This is the most fundamental entropy estimator and serves as the baseline for comparison with other bias-corrected estimators. While it provides an asymptotically unbiased estimate of the true entropy, it can exhibit significant bias for small sample sizes, particularly when the number of unique values is large relative to the sample size.

```{code-cell}
import infomeasure as im
import numpy as np

data = [0, 1, 0, 1, 0, 1, 0, 1]
im.entropy(data, approach="discrete", base=2)
```

In this example data, each state of $0$ or $1$ has a probability of $0.5$, resulting in entropy of
$H(X) = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = -\log_2\left(\tfrac{1}{2}\right) = \log_2 2 = 1$ bit.

#### Miller-Madow Estimator
The Miller-Madow estimator applies a bias correction to the maximum likelihood estimate by adding the correction term $(K-1)/(2N)$, where $K$ is the number of unique values and $N$ is the sample size. This provides improved estimates for small sample sizes.

$$
\hat{H}_{\text{MM}} = \hat{H}_{\text{MLE}} + \frac{K - 1}{2N}
$$

where $\hat{H}_{\text{MLE}}$ is the maximum likelihood entropy estimate, $K$ is the number of unique values in the data, and $N$ is the number of observations. The correction term $(K-1)/(2N)$ provides a simple first-order bias correction that becomes negligible for large sample sizes.

```{code-cell}
# Miller-Madow bias-corrected entropy
data_small = [0, 1, 2, 0, 1, 0]  # Small sample with bias
entropy_discrete = im.entropy(data_small, approach="discrete", base=2)
entropy_mm = im.entropy(data_small, approach="miller_madow", base=2)
print(f"Discrete (MLE): {entropy_discrete:.4f} bits")
print(f"Miller-Madow: {entropy_mm:.4f} bits")
```

### Bias-Corrected Estimators

#### Grassberger Estimator
The Grassberger estimator uses finite sample corrections with the digamma function, providing bias-corrected entropy estimates through count-based corrections {cite:p}`grassbergerFiniteSampleCorrections1988,grassbergerEntropyEstimatesInsufficient2008`.

$$
\hat{H}_{\text{Grassberger}} = \sum_i \frac{n_i}{N} \left(\log(N) - \psi(n_i) - \frac{(-1)^{n_i}}{n_i + 1}  \right)
$$

where $n_i$ are the counts for each unique value, $N$ is the total number of observations, and $\psi$ is the digamma function. This estimator provides sophisticated finite sample corrections that account for the discrete nature of count data and the bias inherent in small sample entropy estimation.

```{code-cell}
entropy_grassberger = im.entropy(data_small, approach="grassberger", base=2)
print(f"Grassberger: {entropy_grassberger:.4f} bits")
```

#### Shrinkage Estimator
The James-Stein shrinkage estimator applies shrinkage to probability estimates before computing entropy, reducing bias in small sample scenarios through regularization toward uniform distribution {cite:p}`hausserEntropyInferenceJamesStein2009`.

The shrinkage probabilities are calculated as:

$$
\hat{p}_x^{\text{SHR}} = \lambda t_x + (1 - \lambda) \hat{p}_x^{\text{ML}}
$$

where $\hat{p}_x^{\text{ML}}$ are the maximum likelihood probability estimates, $t_x = 1/K$ is the uniform target distribution, and the shrinkage parameter $\lambda$ is given by:

$$
\lambda = \frac{ 1 - \sum_{x=1}^{K} (\hat{p}_x^{\text{ML}})^2}{(N-1) \sum_{x=1}^K (t_x - \hat{p}_x^{\text{ML}})^2}
$$

The entropy is then computed using these shrinkage-corrected probabilities: $\hat{H}_{\text{SHR}} = -\sum \hat{p}_x^{\text{SHR}} \log \hat{p}_x^{\text{SHR}}$. This approach provides regularization toward a uniform distribution, which can be particularly effective for small independent samples.

```{code-cell}
entropy_shrink = im.entropy(data_small, approach="shrink", base=2)
print(f"Shrinkage: {entropy_shrink:.4f} bits")
```

### Coverage-Based Estimators

#### Chao-Shen Estimator
The Chao-Shen estimator accounts for unobserved species through coverage estimation using singleton counts, providing improved estimates for incomplete sampling scenarios {cite:p}`chaoNonparametricEstimationShannons2003`.

$$
\hat{H}_{\text{CS}} = - \sum_{i=1}^{K} \frac{\hat{p}_i^{\text{CS}} \log \hat{p}_i^{\text{CS}}}{1 - (1 - \hat{p}_i^{\text{ML}} C)^N}
$$

where

$$
\hat{p}_i^{\text{CS}} = C \cdot \hat{p}_i^{\text{ML}}
$$

and $C = 1 - \frac{f_1}{N}$ is the estimated coverage, $f_1$ is the number of singletons (species observed exactly once), $\hat{p}_i^{\text{ML}}$ is the maximum likelihood probability estimate, $N$ is the sample size, and $K$ is the number of observed species. The Chao-Shen estimator provides a bias-corrected estimate of Shannon entropy that accounts for unobserved species through coverage estimation.

```{code-cell}
entropy_cs = im.entropy(data_small, approach="chao_shen", base=2)
print(f"Chao-Shen: {entropy_cs:.4f} bits")
```

#### Chao-Wang-Jost Estimator
An advanced bias-corrected estimator that uses coverage estimation based on singleton and doubleton counts to account for unobserved species {cite:p}`chaoEntropySpeciesAccumulation2013`.

The Chao-Wang-Jost estimator extends the Chao-Shen approach by using both singleton ($f_1$) and doubleton ($f_2$) counts for more sophisticated coverage estimation:

$$
\hat{C}_{\text{CWJ}} = 1 - \frac{f_1}{N} \cdot \frac{(N-1)f_1}{(N-1)f_1 + 2f_2}
$$

This improved coverage estimate $\hat{C}_{\text{CWJ}}$ is then used in a similar framework to Chao-Shen, providing better bias correction for incomplete sampling scenarios where both singletons and doubletons provide information about unobserved species.

```{code-cell}
entropy_cwj = im.entropy(data_small, approach="chao_wang_jost", base=2)
print(f"Chao-Wang-Jost: {entropy_cwj:.4f} bits")
```

### Bayesian Estimators

#### Bayesian Estimator
A Bayesian entropy estimator with concentration parameter α supporting multiple prior choices (Jeffrey, Laplace, Schurmann-Grassberger, Minimax) for improved entropy estimation with prior knowledge incorporation.

The Bayesian probabilities are calculated as:

$$
p_k^{\text{Bayes}} = \frac{n_k + \alpha}{N + K \alpha}
$$

where $n_k$ is the count of symbol $k$, $N$ is the total number of observations, $K$ is the support size (number of unique symbols), and $\alpha$ is the concentration parameter of the Dirichlet prior. The entropy is then computed as $\hat{H}_{\text{Bayes}} = -\sum p_k^{\text{Bayes}} \log p_k^{\text{Bayes}}$.

**Concentration Parameter Choices:**

- **Jeffreys Prior** ($\alpha = 0.5$): Non-informative prior that is invariant under reparameterization
- **Laplace Prior** ($\alpha = 1.0$): Uniform prior that adds one pseudocount to each symbol
- **Schürmann-Grassberger Prior** ($\alpha = 1/K$): Adaptive prior that scales with the alphabet size
- **Minimax Prior** ($\alpha = \sqrt{N}/K$): Minimizes the maximum expected loss

```{code-cell}
# Bayesian estimator with Jeffrey prior (α = 0.5)
entropy_bayes = im.entropy(data_small, approach="bayes", alpha=0.5, base=2)
print(f"Bayesian (Jeffrey): {entropy_bayes:.4f} bits")

# Bayesian estimator with Laplace prior (α = 1.0)
entropy_bayes_laplace = im.entropy(data_small, approach="bayes", alpha=1.0, base=2)
print(f"Bayesian (Laplace): {entropy_bayes_laplace:.4f} bits")
```

#### NSB (Nemenman-Shafee-Bialek) Estimator
Provides Bayesian estimates of Shannon entropy for discrete data using numerical integration. Particularly effective for undersampled data where traditional estimators may be biased {cite:p}`nemenmanEntropyInferenceRevisited2002`.

The NSB estimate is computed as:

$$
\hat{H}^{\text{NSB}} = \frac{ \int_0^{\ln(K)} d\xi \, \rho(\xi, \mathbf{n}) \langle H^m \rangle_{\beta (\xi)}  }
                            { \int_0^{\ln(K)} d\xi \, \rho(\xi\mid \mathbf{n})}
$$

where

$$
\rho(\xi \mid \mathbf{n}) =
    \mathcal{P}(\beta (\xi)) \frac{ \Gamma(\kappa(\xi))}{\Gamma(N + \kappa(\xi))}
    \prod_{i=1}^K \frac{\Gamma(n_i + \beta(\xi))}{\Gamma(\beta(\xi))}.
$$

The algorithm uses numerical integration to compute the Bayesian posterior over possible entropy values, providing a principled approach to entropy estimation that accounts for sampling uncertainty. This estimator is computationally intensive but provides excellent performance for correlated and undersampled data.

```{code-cell}
entropy_nsb = im.entropy(data_small, approach="nsb", base=2)
print(f"NSB: {entropy_nsb:.4f} bits")
```

### Specialized Estimators

#### ANSB Estimator
The Asymptotic NSB entropy estimator is designed for extremely undersampled discrete data where the number of unique values K is comparable to the sample size N {cite:p}`nemenmanEntropyInformationNeural2004`.

```{code-cell}
# ANSB for undersampled data
undersampled_data = [1, 1, 2, 2, 3, 3]
entropy_ansb = im.entropy(undersampled_data, approach="ansb", base=2)
print(f"ANSB: {entropy_ansb:.4f} bits")
```

#### Zhang Estimator
Uses the recommended definition from Grabchak et al. with fast calculation approach and bias correction through sophisticated probability weighting {cite:p}`grabchakAuthorshipAttributionUsing2013,lozanoFastCalculationEntropy2017`.

```{code-cell}
entropy_zhang = im.entropy(data_small, approach="zhang", base=2)
print(f"Zhang: {entropy_zhang:.4f} bits")
```

#### Bonachela Estimator
Designed for small data sets, providing a compromise between low bias and small statistical errors for short data series {cite:p}`bonachelaEntropyEstimatesSmall2008`.

```{code-cell}
entropy_bonachela = im.entropy(data_small, approach="bonachela", base=2)
print(f"Bonachela: {entropy_bonachela:.4f} bits")
```

## Estimator Comparison

```{code-cell}
# Compare multiple estimators on the same data
estimators = ["discrete", "miller_madow", "grassberger", "shrink", "chao_shen", "nsb"]
results = {}

for estimator in estimators:
    result = im.entropy(data_small, approach=estimator, base=2)
    results[estimator] = result
    print(f"{estimator:15}: {result:.4f} bits")
```

## When to Use Each Estimator

- **Discrete (MLE)**: Large samples (N ≥ 1000) where bias is less of a concern
- **Miller-Madow**: Simple bias correction for medium to large samples
- **NSB**: Best for correlated/Markovian data and small samples
- **Shrinkage**: Good for independent data with small samples
- **Grassberger**: Finite sample corrections with mathematical rigor
- **Chao-Shen/Chao-Wang-Jost**: When unobserved species are suspected
- **Bayesian**: When prior knowledge about the distribution is available
- **ANSB**: Extremely undersampled data
- **Zhang/Bonachela**: Specialized scenarios with specific bias-variance requirements

For detailed guidance on estimator selection, see the {ref}`estimator_selection_guide`.

## Working with Estimator Objects

Some discrete entropy estimators provide access to the underlying distribution and support local values calculation.

```{code-cell}
# Create an estimator object for detailed analysis
data = [1, 2, 3, 1, 2, 1, 2, 3]
est = im.estimator(data, measure="h", approach="discrete", base=2)

# Access the result and distribution
print(f"Entropy: {est.result():.4f} bits")
print(f"Distribution: {est.data[0].distribution_dict}")
print(f"Probabilities sum to: {sum(est.data[0].probabilities):.1f}")
```

### Local Values

All estimators support {ref}`local values <local entropy>` calculation, which provides the information content for each individual observation.

```{code-cell}
from numpy import mean

# Calculate local entropy values
local_values = est.local_vals()
print(f"Local values: {local_values}")
print(f"Mean of local values: {mean(local_values):.4f} bits")
print(f"Global entropy: {est.result():.4f} bits")
print(f"Values are equal: {abs(est.result() - mean(local_values)) < 1e-10}")
```

The mean of the local values $\langle h(x) \rangle$ equals the global entropy $H(X)$, verifying the mathematical relationship.

## Implementation Details

The discrete entropy estimators are implemented in the following classes:

- {py:class}`DiscreteEntropyEstimator <infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator>` - Basic MLE estimator
- {py:class}`MillerMadowEntropyEstimator <infomeasure.estimators.entropy.miller_madow.MillerMadowEntropyEstimator>` - Miller-Madow bias correction
- {py:class}`GrassbergerEntropyEstimator <infomeasure.estimators.entropy.grassberger.GrassbergerEntropyEstimator>` - Grassberger finite sample corrections
- {py:class}`ShrinkEntropyEstimator <infomeasure.estimators.entropy.shrink.ShrinkEntropyEstimator>` - James-Stein shrinkage
- {py:class}`ChaoShenEntropyEstimator <infomeasure.estimators.entropy.chao_shen.ChaoShenEntropyEstimator>` - Chao-Shen coverage estimation
- {py:class}`ChaoWangJostEntropyEstimator <infomeasure.estimators.entropy.chao_wang_jost.ChaoWangJostEntropyEstimator>` - Advanced coverage estimation
- {py:class}`BayesEntropyEstimator <infomeasure.estimators.entropy.bayes.BayesEntropyEstimator>` - Bayesian estimation
- {py:class}`NsbEntropyEstimator <infomeasure.estimators.entropy.nsb.NsbEntropyEstimator>` - NSB Bayesian estimation
- {py:class}`AnsbEntropyEstimator <infomeasure.estimators.entropy.ansb.AnsbEntropyEstimator>` - Asymptotic NSB
- {py:class}`ZhangEntropyEstimator <infomeasure.estimators.entropy.zhang.ZhangEntropyEstimator>` - Zhang estimation
- {py:class}`BonachelaEntropyEstimator <infomeasure.estimators.entropy.bonachela.BonachelaEntropyEstimator>` - Bonachela estimation

All estimators are part of the {py:mod}`infomeasure.estimators.entropy <infomeasure.estimators.entropy>` module.
