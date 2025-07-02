---
file_format: mystnb
kernelspec:
  name: python3
---

(estimator_selection_guide)=
# Estimator Selection Guide

This guide helps you choose the most appropriate estimator and measure for your data and analysis goals. The infomeasure package offers many estimator-measure combinations, and we're always happy to receive pull requests for additional implementations.

**Future Development**: Variational MI estimators (DV, BA, TUBA, $I_α$) are planned for large datasets using stochastic variational inference, as outlined in the changelog.

## How to Use This Guide

Instead of navigating complex diagrams, this guide uses a question-and-answer approach. Start with the first question below, and follow the links to find the most suitable estimator for your specific needs.

## Start Here: What Type of Data Do You Have?

**→ [Discrete data](#discrete-data-selection) (categorical, integer values, finite alphabet)**
- Examples: DNA sequences, text, survey responses, discrete time series
- Go to: [Discrete Data Selection](#discrete-data-selection)

**→ [Continuous data](#continuous-data-selection) (real-valued, measurements)**
- Examples: sensor readings, financial data, physical measurements
- Go to: [Continuous Data Selection](#continuous-data-selection)

**→ [Time series data](#time-series-data-selection) (ordinal/symbolic/permutation approach)**
- Examples: continuous time series, sequential measurements, temporal data
- Special approach: Converts continuous time series to ordinal patterns
- Go to: [Time Series Data Selection](#time-series-data-selection)

**→ [Not sure about your data type?](#data-type-help)**
- Go to: [Data Type Help](#data-type-help)

(discrete-data-selection)=
## Discrete Data Selection

```{admonition} Research Foundation
:class: note

The discrete estimator recommendations in this guide are based on the comprehensive meta-analysis in {cite:p}`degregorioEntropyEstimatorsMarkovian2024`, which evaluated the performance of discrete entropy estimators which have been added version `0.5.0`.
This study provides the empirical foundation for our recommendations on discrete entropy estimators.

The Bonachela and Zhang estimators are also available in `infomeasure` but were not included in the comprehensive meta-analysis in {cite:p}`degregorioEntropyEstimatorsMarkovian2024`. These estimators were added based on their theoretical contributions and are described below with recommendations based on their documented characteristics.
```

Before continuing to the next question we want to note that all discrete estimators in `infomeasure` can calculate multiple information measures, not just entropy.
While discrete entropy estimators excel at entropy estimation, they can compute:
- **Entropy H(X)** - their primary strength
- **Mutual Information I(X;Y)** - statistical dependence between variables
- **Conditional Mutual Information I(X;Y|Z)** - dependence controlling for other variables
- **Transfer Entropy TE(X→Y)** - directed information transfer
- **Conditional Transfer Entropy CTE(X→Y|Z)** - transfer entropy controlling for other variables


### What is your sample size?

**→ [Small sample (N < 100)](#small-discrete-samples)**
- You have fewer than 100 data points
- Go to: [Small Discrete Samples](#small-discrete-samples)

**→ [Medium sample (100 ≤ N < 1000)](#medium-discrete-samples)**
- You have between 100 and 1000 data points
- Go to: [Medium Discrete Samples](#medium-discrete-samples)

**→ [Large sample (N ≥ 1000)](#large-discrete-samples)**
- You have 1000 or more data points
- Go to: [Large Discrete Samples](#large-discrete-samples)

**→ [Special cases](#specialized-discrete-estimators)**
- You have prior knowledge or extremely undersampled data
- Go to: [Specialized Discrete Estimators](#specialized-discrete-estimators)

(small-discrete-samples)=
### Small Discrete Samples (N < 100)

**Are your data points correlated or independent?**

**→ Correlated/Sequential data (e.g., time series, Markov chains)**
- **Recommended**: **NSB (Nemenman-Shafee-Bialek)** - `approach="nsb"`
- **Why**: Lowest mean squared error for correlated data, handles bias and variance well
- **Trade-off**: Computationally intensive, requires numerical integration
- **Reference**: {cite:p}`nemenmanEntropyInferenceRevisited2002`

```{code-cell}
import infomeasure as im
import numpy as np

# Example with small, potentially correlated data
data = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]  # Small sample
entropy_nsb = im.entropy(data, approach="nsb")
print(f"NSB Entropy: {entropy_nsb:.4f}")

# NSB can also calculate other measures with discrete data
data_x = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
data_y = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
mi_nsb = im.mutual_information(data_x, data_y, approach="nsb")
print(f"NSB Mutual Information: {mi_nsb:.4f}")
te_nsb = im.transfer_entropy(data_x, data_y, approach="nsb")
print(f"NSB TE: {te_nsb:.4f}")

```

**→ Independent data (e.g., random samples)**
- **Recommended**: **Shrinkage Estimator** - `approach="shrink"` or `approach="js"`
- **Why**: Lowest MSE for independent data, regularization toward uniform distribution
- **Trade-off**: Less effective for correlated data
- **Reference**: {cite:p}`hausserEntropyInferenceJamesStein2009`

```{code-cell}
# Good for independent, small samples
entropy_shrink = im.entropy(data, approach="shrink")
print(f"Shrinkage Entropy: {entropy_shrink:.4f}")

# Shrinkage can also calculate transfer entropy with discrete data
te_shrink = im.transfer_entropy(data_x, data_y, approach="shrink")
print(f"Shrinkage Transfer Entropy: {te_shrink:.4f}")
```

**→ Very small samples with balanced probabilities**
- **Recommended**: **Bonachela (Bonachela-Hinrichsen-Muñoz)** - `approach="bonachela"`
- **Why**: Specially designed for short data series, provides compromise between low bias and small statistical errors
- **Best for**: Small datasets where probabilities are not close to zero
- **Trade-off**: Limited theoretical validation compared to NSB
- **Reference**: {cite:p}`bonachelaEntropyEstimatesSmall2008`

```{code-cell}
# Example with very small, balanced data
small_balanced_data = [0, 1, 2, 0, 1, 2, 0, 1]  # Small, balanced sample
entropy_bonachela = im.entropy(small_balanced_data, approach="bonachela")
print(f"Bonachela Entropy: {entropy_bonachela:.4f}")

# Bonachela can also calculate other measures
mi_bonachela = im.mutual_information(data_x, data_y, approach="bonachela")
print(f"Bonachela Mutual Information: {mi_bonachela:.4f}")
```

**→ Incomplete sampling (suspect unobserved states)**
- **Recommended**: **Chao-Shen Estimator** - `approach="chao_shen"` or `approach="cs"`
- **Why**: Accounts for unobserved species using coverage estimation
- **When**: You believe there are states in your data that you haven't observed yet
- **Reference**: {cite:p}`chaoNonparametricEstimationShannons2003`

(medium-discrete-samples)=
### Medium Discrete Samples (100 ≤ N < 1000)

**Do you need sophisticated bias correction?**

**→ Yes, I need advanced bias correction**
- **For correlated data**: **NSB** - `approach="nsb"` (still best choice)
- **For general use**: **Chao-Wang-Jost** - `approach="chao_wang_jost"` or `approach="cwj"`
  - Uses singleton and doubleton counts for coverage estimation
  - Sophisticated bias correction for incomplete sampling
  - **Reference**: {cite:p}`chaoEntropySpeciesAccumulation2013,marconEntropartPackageMeasure2015`

**→ No, simple bias correction is sufficient**
- **Recommended**: **Miller-Madow** - `approach="miller_madow"` or `approach="mm"`
- **Why**: Simple correction term (K-1)/(2N), computationally efficient
- **Alternative**: **Grassberger** - `approach="grassberger"`
  - Finite sample corrections with digamma function
  - Count-based corrections, mathematically principled
  - **Reference**: {cite:p}`grassbergerFiniteSampleCorrections1988,grassbergerEntropyEstimatesInsufficient2008`
- **Alternative for bias correction**: **Zhang** - `approach="zhang"`
  - Uses sophisticated bias correction with cumulative product factors
  - Fast calculation approach for entropy estimation
  - **Reference**: {cite:p}`grabchakAuthorshipAttributionUsing2013,lozanoFastCalculationEntropy2017`

```{code-cell}
# Medium-sized sample with guaranteed singletons and doubletons
np.random.seed(92183)  # For reproducible results
medium_data = np.random.choice([0, 1, 2, 3, 4, 5], size=500, p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05])
# Add some singletons and doubletons explicitly
medium_data = np.concatenate([medium_data, [6], [7, 7]])  # Add singleton 6 and doubleton 7

entropy_mm = im.entropy(medium_data, approach="miller_madow")
entropy_cwj = im.entropy(medium_data, approach="chao_wang_jost")
print(f"Miller-Madow: {entropy_mm:.4f}")
print(f"Chao-Wang-Jost: {entropy_cwj:.4f}")

# Miller-Madow can also calculate conditional mutual information
medium_x = np.random.choice([0, 1], size=500)
medium_y = np.random.choice([0, 1], size=500)
medium_z = np.random.choice([0, 1], size=500)
cmi_mm = im.conditional_mutual_information(medium_x, medium_y, cond=medium_z, approach="miller_madow")
print(f"Miller-Madow Conditional MI: {cmi_mm:.4f}")
```

```{code-cell}
# Zhang estimator for medium samples
entropy_zhang = im.entropy(medium_data, approach="zhang")
print(f"Zhang Entropy: {entropy_zhang:.4f}")

# Zhang can also calculate transfer entropy
te_zhang = im.transfer_entropy(data_x, data_y, approach="zhang")
print(f"Zhang Transfer Entropy: {te_zhang:.4f}")
```

(large-discrete-samples)=
### Large Discrete Samples (N ≥ 1000)

**Do you prioritize speed or bias correction?**

**→ Speed is most important**
- **Recommended**: **Discrete (MLE)** - `approach="discrete"`
- **Why**: Fastest computation, well-understood, bias becomes less important with large samples

**→ Still want some bias correction**
- **Recommended**: **Miller-Madow** - `approach="miller_madow"`
- **Why**: Minimal computational overhead over MLE, simple bias correction

(specialized-discrete-estimators)=
### Specialized Discrete Estimators

**Do you have prior knowledge about your data distribution?**

**→ Yes, I have prior knowledge**
- **Recommended**: **Bayesian Estimator** - `approach="bayes"`
- **Available priors**: Jeffrey, Laplace, Schurmann-Grassberger, Minimax
- **Usage**: Specify prior with `alpha` parameter
- **Reference**: {cite:p}`krichevskyPerformanceUniversalEncoding1981,bayesEssaySolvingProblem1763`

```{code-cell}
# Bayesian with different priors
data = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
entropy_bayes = im.entropy(data, approach="bayes", alpha=0.5)  # Jeffrey prior
print(f"Bayesian Entropy: {entropy_bayes:.4f}")
```

**→ Extremely undersampled data**
- **Recommended**: **ANSB** - `approach="ansb"`
- **When**: Number of unique values is close to sample size
- **Why**: Efficient for undersampled regime
- **Reference**: {cite:p}`nemenmanEntropyInformationNeural2004`

(continuous-data-selection)=
## Continuous Data Selection

### What information measure do you need?

**→ [Entropy H(X)](#continuous-entropy-selection)**
- You want to measure the uncertainty/information content of a single variable
- Go to: [Continuous Entropy Selection](#continuous-entropy-selection)

**→ [Mutual Information I(X;Y)](#continuous-mi-selection)**
- You want to measure statistical dependence between two variables
- Go to: [Continuous MI Selection](#continuous-mi-selection)

**→ [Transfer Entropy TE(X→Y)](#continuous-te-selection)**
- You want to measure directed information transfer between variables
- Go to: [Continuous TE Selection](#continuous-te-selection)

**→ [Other measures](#other-continuous-measures)**
- Conditional mutual information, cross-entropy, etc.
- Go to: [Other Continuous Measures](#other-continuous-measures)

(continuous-entropy-selection)=
### Continuous Entropy Selection

**What are your data characteristics?**

**→ High-dimensional data OR small to medium samples**
- **Recommended**: **Kozachenko-Leonenko (KL)** - `approach="metric"` or `approach="kl"`
- **Why**: No bandwidth selection needed, adapts to local density
- **Best for**: High-dimensional data, small to medium samples
- **Method**: Nearest neighbor approach

```{code-cell}
# Continuous data example
continuous_data = np.random.normal(0, 1, 1000)
entropy_kl = im.entropy(continuous_data, approach="metric")
print(f"KL Entropy: {entropy_kl:.4f}")
```

**→ Low-dimensional data AND large samples**
- **Recommended**: **Kernel Estimator** - `approach="kernel"`
- **Why**: Flexible, well-understood density estimation
- **Trade-off**: Requires bandwidth selection
- **Best for**: Low-dimensional data, large samples

```{code-cell}
entropy_kernel = im.entropy(
    continuous_data, approach="kernel", kernel="box", bandwidth=0.5)
print(f"Kernel Entropy: {entropy_kernel:.4f}")
```

(continuous-mi-selection)=
### Continuous Mutual Information Selection

**Note**: All continuous mutual information estimators support any number of random variables for multivariate mutual information calculation.

**What is your sample size?**

**→ Large samples (efficient computation needed)**
- **Recommended**: **KSG (Kraskov-Stögbauer-Grassberger)** - `approach="ksg"` or `approach="metric"`
- **Why**: Efficient, well-validated for large datasets
- **Best for**: Large samples where computational efficiency matters

**→ Small to medium samples (need control over bandwidth)**
- **Recommended**: **Kernel MI** - `approach="kernel"`
- **Why**: More control over bandwidth selection
- **Best for**: Smaller samples where you can carefully tune parameters

(continuous-te-selection)=
### Continuous Transfer Entropy Selection

**→ Most transfer entropy applications**
- **Recommended**: **Kernel TE** - `approach="kernel"`
- **Why**: Most flexible approach for transfer entropy
- **Usage**: Works well for most continuous transfer entropy applications

(other-continuous-measures)=
### Other Continuous Measures

**→ Conditional Mutual Information I(X;Y|Z)**
- **Recommended**: Use the same estimators as for mutual information
- **Usage**: `im.conditional_mutual_information(X, Y, cond=Z, approach="ksg")`

**→ Cross-Entropy and KL Divergence**
- **Available for**: Estimators with cross-entropy support
- **Usage**: See Kullback-Leibler Divergence and Jensen-Shannon Divergence sections below

(time-series-data-selection)=
## Time Series Data Selection

### Ordinal/Symbolic/Permutation Approach

**→ For all time series analysis applications**
- **Recommended**: **Ordinal Estimator** - `approach="ordinal"`
- **Why**: Converts continuous/discrete time series to ordinal patterns based on relative ordering
- **Best for**: Time series complexity analysis, temporal pattern detection
- **Key parameter**: `embedding_dim` - size of the sliding window for pattern extraction
- **Supports**: Entropy, Mutual Information, Transfer Entropy, and conditional measures

```{code-cell}
import infomeasure as im
import numpy as np

# Example time series data
np.random.seed(666)
time_series = np.random.normal(0, 1, 1000)

# Ordinal entropy with embedding dimension 3
entropy_ordinal = im.entropy(time_series, approach="ordinal", embedding_dim=3)
print(f"Ordinal Entropy: {entropy_ordinal:.4f}")

# Ordinal mutual information between two time series
time_series_2 = np.random.normal(0, 1, 1000)
mi_ordinal = im.mutual_information(time_series, time_series_2, approach="ordinal", embedding_dim=3)
print(f"Ordinal MI: {mi_ordinal:.4f}")

# Ordinal transfer entropy for causal analysis
te_ordinal = im.transfer_entropy(time_series, time_series_2, approach="ordinal", embedding_dim=3)
print(f"Ordinal TE: {te_ordinal:.4f}")
```

**→ Choosing embedding dimension**
- **Small embedding (2-3)**: Captures basic temporal patterns, computationally efficient
- **Medium embedding (4-5)**: More detailed pattern analysis, balanced complexity
- **Large embedding (6+)**: Fine-grained patterns, requires more data

**→ Detailed documentation**
- **Entropy**: See {doc}`entropy/ordinal` for comprehensive guide
- **Mutual Information**: See {doc}`mutual_information/ordinal` for detailed examples
- **Transfer Entropy**: See {doc}`transfer_entropy/ordinal` for causal analysis

(data-type-help)=
## Data Type Help

**Not sure if your data is discrete or continuous?**

**→ Your data is likely DISCRETE if:**
- Values are integers or categories (0, 1, 2, 3, ...)
- Finite number of possible values
- Examples: DNA sequences (A, T, G, C), survey responses (1-5 scale), word counts

**→ Your data is likely CONTINUOUS if:**
- Values are real numbers with decimals
- Infinite number of possible values in a range
- Examples: temperature measurements, stock prices, sensor readings

## Information Measure Selection Guide

### Choose based on your research question:

**→ [Entropy H(X)](#entropy-measure-info)**
- **Question**: "How much uncertainty/information is in my variable?"
- **Use cases**: Data compression, feature selection, complexity analysis
- Go to: [Entropy Measure Info](#entropy-measure-info)

**→ [Mutual Information I(X;Y)](#mutual-information-measure-info)**
- **Question**: "How much do two variables depend on each other?"
- **Use cases**: Feature selection, correlation analysis, independence testing
- Go to: [Mutual Information Measure Info](#mutual-information-measure-info)

**→ [Transfer Entropy TE(X→Y)](#transfer-entropy-measure-info)**
- **Question**: "Does X influence Y over time?"
- **Use cases**: Causality analysis, time series analysis, network inference
- Go to: [Transfer Entropy Measure Info](#transfer-entropy-measure-info)

**→ [Conditional Measures](#conditional-measures-info)**
- **Question**: "How do variables relate when controlling for others?"
- **Use cases**: Partial correlation, confounding variable analysis
- Go to: [Conditional Measures Info](#conditional-measures-info)

**→ Composite Measures**
- **Question**: "How similar/different are two distributions?"
- **Use cases**: Model comparison, distribution similarity
- Go to: Kullback-Leibler Divergence or Jensen-Shannon Divergence sections below

(entropy-measure-info)=
### Entropy H(X)
- **Purpose**: Quantify uncertainty/information content of a single variable
- **Interpretation**: Higher values = more uncertainty/information
- **Units**: Depends on logarithm base (bits for base 2, nats for base e)
- **Range**: 0 to log(K) where K is number of unique values
- **Detailed documentation**: See {ref}`entropy_overview`

(mutual-information-measure-info)=
### Mutual Information I(X;Y)
- **Purpose**: Measure statistical dependence between variables
- **Interpretation**: 0 = independent, higher values = more dependent
- **Symmetric**: I(X;Y) = I(Y;X)
- **Range**: 0 to min(H(X), H(Y))
- **Detailed documentation**: See {ref}`mutual_information_overview`

(transfer-entropy-measure-info)=
### Transfer Entropy TE(X→Y)
- **Purpose**: Directed information transfer from X to Y
- **Interpretation**: How much X's past helps predict Y's future
- **Asymmetric**: TE(X→Y) ≠ TE(Y→X) in general
- **Range**: 0 to H(Y)
- **Detailed documentation**: See {ref}`transfer_entropy_overview`

(conditional-measures-info)=
### Conditional Measures
- **Conditional Mutual Information I(X;Y|Z)**: Dependence between X and Y given Z
- **Conditional Entropy H(X|Y)**: Uncertainty in X given knowledge of Y
- **Conditional Transfer Entropy**: Transfer entropy controlling for other variables
- **Detailed documentation**: See {ref}`cond_mi_overview` and {ref}`cond_te_overview`

### Kullback-Leibler Divergence D_KL(P||Q)
- **Purpose**: Information lost when Q approximates P
- **Use cases**: Model selection, distribution comparison
- **Available for**: Estimators with cross-entropy support
- **Usage**: `im.kullback_leibler_divergence(P, Q, approach="discrete")`
- **Detailed documentation**: See {doc}`KLD`

### Jensen-Shannon Divergence JSD(P,Q)
- **Purpose**: Symmetric measure of distribution similarity
- **Use cases**: Clustering, distribution comparison
- **Available for**: Bayes, Shrinkage, and pre-`v0.5.0` estimators
- **Usage**: `im.jensen_shannon_divergence(P, Q, approach="bayes")`
- **Detailed documentation**: See {doc}`JSD`

## Performance Considerations

**Need to choose between estimators with similar capabilities?**

### Computational Complexity

| Estimator | Complexity | Speed | Memory   | Best for                                    |
|-----------|------------|-------|----------|---------------------------------------------|
| Discrete | O(N) | Fastest | Minimal  | Large samples                               |
| Miller-Madow | O(N) | Fastest | Minimal  | General use                                 |
| Grassberger | O(N) | Fast | Minimal  | Mathematical rigor                          |
| Shrinkage | O(N) | Fast | Minimal  | Small independent samples                   |
| Bonachela | O(N) | Fast | Minimal  | Very small balanced samples                 |
| Zhang | O(N) | Fast | Moderate | Medium samples with bias correction         |
| Chao-Shen | O(N) | Fast | Minimal  | Incomplete sampling                         |
| Chao-Wang-Jost | O(N) | Moderate | Minimal  | Advanced bias correction                    |
| Bayesian | O(N) | Fast | Minimal  | Prior knowledge                             |
| NSB | O(N log N) | Slow | Moderate | Correlated data                             |
| ANSB | O(N log N) | Moderate | Moderate | Undersampled regime                         |
| Ordinal | O(N) | Fast | Minimal  | Time series analysis, continuous & discrete |
| Renyi | O(N log N) | Moderate | Moderate | Continuous, generalized entropy             |
| Tsallis | O(N log N) | Moderate | Moderate | Continuous, non-extensive systems           |
| Kernel | O(N²) | Slow | High     | Continuous, low-dim                         |
| KSG | O(N log N) | Moderate | Moderate | Continuous, large samples                   |
| Kozachenko-Leonenko | O(N log N) | Moderate | Moderate | Continuous, high-dim                        |

### Statistical Properties for Discrete Entropy Estimators

**Note**: These properties refer specifically to discrete entropy estimation based on {cite:p}`degregorioEntropyEstimatorsMarkovian2024`. The Bonachela and Zhang estimators were not included in this meta-analysis but are available based on their theoretical contributions. The continuous estimators and other measures offered by infomeasure are not covered in this analysis.

- **Lowest Bias**: NSB, Chao-Wang-Jost
- **Lowest Variance**: MLE (Discrete), Miller-Madow
- **Best MSE**: NSB (correlated data), Shrinkage (independent data)
- **Most Robust**: Miller-Madow, Grassberger
- **Specialized Use Cases**: Bonachela (very small balanced samples), Zhang (medium samples with bias correction)

## Practical Examples

### Example 1: Time Series Analysis (Correlated Data)
```{code-cell}
# Potentially correlated time series
time_series = np.random.choice([0, 1], size=200, p=[0.7, 0.3])
# Add some temporal correlation
for i in range(1, len(time_series)):
    if np.random.random() < 0.3:  # 30% chance to copy previous
        time_series[i] = time_series[i-1]

# Use NSB for correlated data
entropy_ts = im.entropy(time_series, approach="bonachela")
print(f"Time series entropy (Bonachela): {entropy_ts:.4f}")
```

### Example 2: Feature Selection (Independent Data)
```{code-cell}
# Independent features for classification
features = np.random.randint(0, 5, size=(1000, 3))
target = np.random.randint(0, 2, size=1000)

# Use Miller-Madow for medium-sized independent data
mi_values = []
for i in range(features.shape[1]):
    mi = im.mutual_information(features[:, i], target, approach="miller_madow")
    mi_values.append(mi)

print(f"MI values: {mi_values}")
```

### Example 3: Continuous Data Analysis
```{code-cell}
# High-dimensional continuous data
X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 500)

# Use KL for continuous entropy
entropy_cont = im.entropy(X[:, 0], approach="metric")
# Use KSG for mutual information
mi_cont = im.mutual_information(X[:, 0], X[:, 1], approach="ksg")

print(f"Continuous entropy: {entropy_cont:.4f}")
print(f"Continuous MI: {mi_cont:.4f}")
```

(example-time-lag-te-pval)=
### Example 4: Time Lag Selection for Transfer Entropy with _P_-value Evaluation
```{code-cell}
# Generate time series data with known causal relationship
np.random.seed(777)  # For reproducible results
n_samples = 200

# Create source time series
source = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])

# Create destination time series with causal influence from source (lag=3 is optimal)
dest = np.zeros(n_samples, dtype=int)
dest[0] = np.random.choice([0, 1, 2])  # Random initial value

for i in range(1, n_samples):
    if i >= 3:  # True causal lag is 3
        # 20% chance to be influenced by source[i-3], 80% random
        if np.random.random() < 0.2:
            dest[i] = source[i-3]
        else:
            dest[i] = np.random.choice([0, 1, 2])
    else:
        dest[i] = np.random.choice([0, 1, 2])

# Test different time lags (1 to 5) and evaluate p-values
print("Testing Transfer Entropy with different time lags:")
print("Lag\tTE Value\tP-value")
print("-" * 32)

lag_results = []
for lag in range(1, 6):
    # Create TE estimator with specific time lag
    te_estimator = im.estimator(
        source, dest,
        measure="transfer_entropy",
        approach="discrete",
        prop_time=lag
    )

    # Get TE value
    te_value = te_estimator.result()

    # Perform statistical test to get p-value
    stat_result = te_estimator.statistical_test(n_tests=100, method="permutation_test")
    p_value = stat_result.p_value

    lag_results.append((lag, te_value, p_value))
    print(f"{lag}\t{te_value:.4f}\t\t{p_value:.4f}")

# Find the lag with the best (lowest) p-value
best_lag, best_te, best_p = min(lag_results, key=lambda x: x[2])

print(f"\nBest time lag: {best_lag}")
print(f"TE value at best lag: {best_te:.4f}")
print(f"Best p-value: {best_p:.4f}")

# Additional analysis: show confidence interval for the best lag
best_estimator = im.estimator(
    source, dest,
    measure="transfer_entropy",
    approach="discrete",
    prop_time=best_lag
)
best_stat_result = best_estimator.statistical_test(n_tests=1000, method="permutation_test")
ci_95 = best_stat_result.percentile([2.5, 97.5])
print(f"95% Confidence Interval for best lag: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
```

## Quick Decision Summary

### Simple Decision Tree

This decision tree helps you choose the appropriate information-theoretic estimator based on your data characteristics and analysis goals. The diagram provides a systematic approach to selecting between different entropy, mutual information, and transfer entropy estimators.
The diagram below is **zoomable** - use your mouse wheel or pinch gestures to zoom in/out for better readability of the detailed decision paths.


```{mermaid}
:zoom:
flowchart TD
    A(What type of data?) --> B[Discrete]
    A --> C[Continuous]
    A --> D[Time Series]

    B --> E(Small sample<br/>N < 100?)
    B --> F(Medium sample<br/>100 ≤ N < 1000?)
    B --> G(Large sample<br/>N ≥ 1000?)

    E --> H(Correlated?)
    E --> I(Independent?)
    H --> J[NSB]
    I --> K[Shrinkage<br/>or Bonachela]

    F --> L[Miller-Madow,<br/>Zhang, or NSB]
    G --> M[Discrete<br/>or Miller-Madow]

    C --> N(What measure?)
    N --> O["Entropy H(X)"]
    N --> P["Mutual Information I(X;Y)"]
    N --> Q["Transfer Entropy TE(X→Y)"]
    N --> R[Other measures]

    O --> S(High-dimensional or<br/>small/medium samples?)
    O --> T(Low-dimensional and<br/>large samples?)
    S --> U[Kozachenko-Leonenko]
    T --> V[Kernel]

    P --> W(Large samples?)
    P --> X(Small/medium samples?)
    W --> Y[KSG]
    X --> Z[Kernel]

    Q --> AA[Kernel TE<br/>Most flexible approach]

    R --> BB[Use same estimators<br/>as for MI/Entropy<br/>with appropriate syntax]

    D --> CC[Ordinal/Symbolic<br/>Permutation Approach]

    %% Styling for question nodes
    classDef questionStyle fill:#e1e1e1,stroke:#999,stroke-width:2px,color:#000
    class A,E,F,G,H,I,N,S,T,W,X questionStyle

    %% Styling for time series node
    classDef timeSeriesStyle fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000
    class CC timeSeriesStyle
```

### Key Recommendations

| Scenario | Recommended Estimator | Approach String | Alternative |
|----------|----------------------|-----------------|-------------|
| **Small discrete, correlated** | NSB | `"nsb"` | Chao-Wang-Jost |
| **Small discrete, independent** | Shrinkage | `"shrink"` | Chao-Shen |
| **Very small discrete, balanced** | Bonachela | `"bonachela"` | Shrinkage |
| **Medium discrete, general** | Miller-Madow | `"miller_madow"` | Grassberger |
| **Medium discrete, advanced** | Chao-Wang-Jost | `"chao_wang_jost"` | Zhang |
| **Medium discrete, bias correction** | Zhang | `"zhang"` | Miller-Madow |
| **Large discrete, speed priority** | Discrete (MLE) | `"discrete"` | Miller-Madow |
| **Large discrete, bias correction** | Miller-Madow | `"miller_madow"` | Grassberger |
| **Prior knowledge available** | Bayesian | `"bayes"` | - |
| **Extremely undersampled** | ANSB | `"ansb"` | NSB |
| **Continuous entropy, high-dim** | Kozachenko-Leonenko | `"metric"` | - |
| **Continuous entropy, low-dim** | Kernel | `"kernel"` | Kozachenko-Leonenko |
| **Continuous MI, large samples** | KSG | `"ksg"` | - |
| **Continuous MI, small samples** | Kernel | `"kernel"` | KSG |
| **Continuous TE** | Kernel | `"kernel"` | - |
| **Time series analysis** | Ordinal | `"ordinal"` | - |
| **When in doubt** | NSB (discrete) or KL (continuous) | `"nsb"` / `"metric"` | Miller-Madow / Kernel |

### General Principles

1. **For correlated/temporal data**: Always prefer NSB or Chao-Wang-Jost
2. **For independent data**: Shrinkage (small N) or Miller-Madow (medium/large N)
3. **For computational efficiency**: Discrete, Miller-Madow, or Grassberger
4. **For theoretical rigor**: NSB, Grassberger, or Bayesian approaches
5. **For continuous data**: KL/KSG for most cases, Kernel for specialized needs
6. **For incomplete sampling**: Chao-Shen, Chao-Wang-Jost, or NSB
7. **For time series analysis**: Ordinal approach converts continuous time series to ordinal patterns

## Time Lag Selection for Transfer Entropy and Mutual Information

### Choosing Optimal Time Lags

Computing transfer entropy and mutual information with temporal data requires selecting appropriate time lags (delays/offsets). The choice of time lag is crucial for:

- **Transfer Entropy**: Determining the delay between cause and effect
- **Mutual Information**: Finding optimal temporal relationships between variables

### Manual Selection

The infomeasure package allows manual time lag selection through the `prop_time` or `offset` parameters:

```python
# Transfer entropy with manual time lag
te_result = im.transfer_entropy(source, dest, approach="kernel", prop_time=5)

# Mutual information with offset
mi_result = im.mutual_information(x, y, approach="kernel", offset=3)
```

### Integration with IDTxl

For systematic lag optimization, a manual loop can also suffice for finding the best lag, but [IDTxl](https://github.com/pwollstadt/IDTxl) can be used to determine optimal time lags for transfer entropy and mutual information analysis. Additionally, infomeasure estimators can be used with IDTxl's [MPI support](https://pwollstadt.github.io/IDTxl/html/idtxl_estimators.html#mpi-estimators-cpu) through the `MPIEstimator` wrapper.

To integrate infomeasure estimators with IDTxl, the infomeasure output needs to be wrapped into a child class of IDTxl's abstract [`Estimator` class](https://github.com/pwollstadt/IDTxl/blob/master/idtxl/estimator.py#L111), which requires implementing methods like `estimate()`, `is_parallel()`, and `is_analytic_null_estimator()`.

## Additional Information

**Note**: The infomeasure package offers many more estimator-measure combinations than covered in this guide. We're always happy to receive pull requests for additional implementations or improvements to existing ones.

**For more details**: See the individual estimator documentation pages and the comprehensive analysis in {cite:p}`degregorioEntropyEstimatorsMarkovian2024`.
