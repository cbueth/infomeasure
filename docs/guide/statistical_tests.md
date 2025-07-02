---
file_format: mystnb
kernelspec:
  name: python3
---

# Statistical Tests

Statistical testing is a fundamental component of information-theoretic analysis, allowing researchers to assess the significance of observed information measures and distinguish meaningful relationships from random fluctuations. The `infomeasure` package provides comprehensive statistical testing capabilities through hypothesis testing, enabling robust statistical inference for entropy, mutual information, and transfer entropy measures.

## Overview

Statistical tests in `infomeasure` help answer the critical question: *Is the observed information measure significantly different from what would be expected by chance?* This is essential for:

- **Validating information-theoretic relationships**: Determining if observed dependencies are statistically meaningful
- **Controlling for multiple comparisons**: Assessing significance when testing many relationships
- **Quantifying uncertainty**: Providing confidence intervals and effect sizes
- **Comparing different estimation methods**: Evaluating the statistical properties of various estimators

The package implements two primary statistical testing approaches:
1. **Permutation tests** (default): Non-parametric tests that shuffle data embeddings to break up temporal patterns
2. **Bootstrap tests**: Resampling methods that generate null distributions through sampling with replacement

## Theoretical Background

### Hypothesis Testing Framework

Statistical tests in `infomeasure` follow the classical hypothesis testing framework:

- **Null Hypothesis (H₀)**: The observed information measure is not different from random (no significant relationship)
- **Alternative Hypothesis (H₁)**: The observed information measure indicates a significant relationship
- **Test Statistic**: The computed information measure (entropy, mutual information, transfer entropy)
- **Null Distribution**: Distribution of the test statistic under the null hypothesis
- **p-value**: Probability of observing a test statistic as extreme or more extreme than the observed value, assuming H₀ is true

For both types of hypothesis tests, when using subsampled data, not the input data is shuffled or resampled, but the embedding itself.

### Permutation Tests

Permutation tests are non-parametric statistical tests that generate the null distribution by permuting (shuffling) the data. The key principle is that under the null hypothesis of no relationship, the joint distribution of variables should be the same as the product of their marginal distributions.

For mutual information, the null hypothesis assumes independence between variables X and Y. Data from one variable (typically the source) is randomly permuted, which breaks any temporal or structural dependencies while preserving marginal distributions. The mutual information is then recalculated for each permutation to build the null distribution.

For transfer entropy, the null hypothesis assumes no causal influence from source to target. Source data is permuted to eliminate causal relationships while target dynamics and any conditioning variables remain intact. Transfer entropy is recalculated for each permutation to assess the significance of the observed causal relationship.

Permutation tests offer several advantages, including requiring no distributional assumptions, providing exact control of Type I error rates, preserving marginal distributions of the data, and being appropriate for complex, non-linear relationships.

### Bootstrap Tests

Bootstrap tests generate the null distribution through resampling with replacement. This approach is particularly useful when the sample size is limited or when you want to assess the variability of the estimator itself.

The bootstrap procedure involves resampling data points with replacement to create bootstrap samples, calculating the information measure for each bootstrap sample, and building the null distribution from these bootstrap statistics. This method provides insight into estimator variability, is particularly useful for small sample sizes, and can be combined with bias correction techniques to improve the accuracy of statistical inference.

## Statistical Test Results

The {func}`~infomeasure.estimators.mixins.StatisticalTestingMixin.statistical_test` method returns a comprehensive {class}`~infomeasure.utils.data.StatisticalTestResult` object containing:

### Core Statistics

The result object provides essential statistical measures including the `p-value`, which represents the proportion of test values greater than the observed value, and the `t-score`, a standardized test statistic calculated as (observed - null_mean) / null_std. The `observed_value` contains the original computed information measure, while `null_mean` and `null_std` provide the mean and standard deviation of the null distribution generated through the resampling procedure.

### Additional Information

The result also contains comprehensive metadata about the statistical test procedure. The `test_values` array holds all values from the resampling procedure, `n_tests` records the number of permutations or bootstrap samples performed, and `method` indicates the statistical test method used ("permutation_test" or "bootstrap").

### Advanced Analysis Methods

The `StatisticalTestResult` object provides methods for detailed statistical analysis:

```python
# Calculate percentiles of the null distribution
median = result.percentile(50)
quartiles = result.percentile([25, 75])

# Compute confidence intervals
ci_95 = result.confidence_interval(95)
ci_99 = result.confidence_interval(99)
```

## Configuration and Usage

### Global Configuration

Statistical testing behavior can be configured globally using the `Config` module:

```{code-cell}
import infomeasure as im
import numpy as np

# Set default statistical test method
im.Config.set("statistical_test_method", "permutation_test")  # or "bootstrap"

# Set default number of tests
im.Config.set("statistical_test_n_tests", 1000)

# Check current settings
print(f"Method: {im.Config.get('statistical_test_method')}")
print(f"Number of tests: {im.Config.get('statistical_test_n_tests')}")
```

### Basic Usage Example

```{code-cell}
# Generate sample data with moderate relationship
rng = np.random.default_rng(456)
n_samples = 500

# Create moderately correlated data
x = rng.normal(0, 1, n_samples)
y = 0.25 * x + np.sqrt(1 - 0.25**2) * rng.normal(0, 1, n_samples)

# Discretize for discrete estimators
x_discrete = np.digitize(x, bins=np.linspace(-3, 3, 6))
y_discrete = np.digitize(y, bins=np.linspace(-3, 3, 6))

# Create estimator and perform statistical test
est = im.estimator(x_discrete, y_discrete, measure="mi", approach="discrete")

# Use global configuration
result = est.statistical_test()
print(f"Mutual Information: {est.global_val():.4f}")
print(f"p-value: {result.p_value:.4f}")
print(f"t-score: {result.t_score:.4f}")
print(f"Method: {result.method}")
```

### Advanced Usage with Custom Parameters

```{code-cell}
# Override global settings for specific test
result_permutation = est.statistical_test(n_tests=500, method="permutation_test")
result_bootstrap = est.statistical_test(n_tests=500, method="bootstrap")

print("Permutation Test Results:")
print(f"  p-value: {result_permutation.p_value:.4f}")
print(f"  t-score: {result_permutation.t_score:.4f}")

print("Bootstrap Test Results:")
print(f"  p-value: {result_bootstrap.p_value:.4f}")
print(f"  t-score: {result_bootstrap.t_score:.4f}")
```

### Confidence Intervals and Percentiles

```{code-cell}
# Calculate confidence intervals
ci_95 = result.confidence_interval(95)
ci_99 = result.confidence_interval(99)

print(f"95% Confidence Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
print(f"99% Confidence Interval: [{ci_99[0]:.4f}, {ci_99[1]:.4f}]")

# Access specific percentiles
print(f"Median of null distribution: {result.percentile(50):.4f}")
print(f"95th percentile: {result.percentile(95):.4f}")
```

## Transfer Entropy Example

Statistical testing is particularly important for transfer entropy, where establishing causal relationships requires careful statistical validation.
For determining an optimal time lag using the _p_-value,
see {ref}`example-time-lag-te-pval`.


```{code-cell}
# Generate time series with moderate causal relationship
rng_te = np.random.default_rng(2354)
n_time = 500
x_ts = rng_te.normal(0, 1, n_time)
y_ts = np.zeros(n_time)

# Create moderate causal relationship: Y(t) depends on X(t-1)
for t in range(1, n_time):
    y_ts[t] = 0.3 * x_ts[t-1] + np.sqrt(1 - 0.2**2) * rng_te.normal(0, 1)

# Discretize time series
x_discrete = np.digitize(x_ts, bins=np.linspace(-3, 3, 4))
y_discrete = np.digitize(y_ts, bins=np.linspace(-3, 3, 4))

# Test transfer entropy
te_est = im.estimator(x_discrete, y_discrete, measure="te", approach="discrete")
te_result = te_est.statistical_test(n_tests=200)

print(f"Transfer Entropy X→Y: {te_est.global_val():.4f}")
print(f"p-value: {te_result.p_value:.4f}")
print(f"Significance: {'Yes' if te_result.p_value < 0.05 else 'No'} (α = 0.05)")

# Test reverse direction
te_est_reverse = im.estimator(y_discrete, x_discrete, measure="te", approach="discrete")
te_result_reverse = te_est_reverse.statistical_test(n_tests=200)

print(f"Transfer Entropy Y→X: {te_est_reverse.global_val():.4f}")
print(f"p-value: {te_result_reverse.p_value:.4f}")
print(f"Significance: {'Yes' if te_result_reverse.p_value < 0.05 else 'No'} (α = 0.05)")
```

## Supported Measures

Statistical testing is available for mutual information, conditional mutual information, transfer entropy, and conditional transfer entropy estimators. All estimators support permutation tests and a bootstrap method with resampling. The tests evaluate independence hypotheses for mutual information measures and causal influence hypotheses for transfer entropy measures, requiring exactly two variables for mutual information and supporting effective value calculations for transfer entropy.

## Error Handling and Limitations

### Limitations

1. **Computational Cost**: Statistical tests require multiple calculations of the information measure
2. **Sample Size Dependency**: Very small samples may not provide reliable statistical inference
3. **Multiple Comparisons**: Testing many relationships requires correction for multiple comparisons
4. **Assumption Violations**: Permutation tests assume exchangeability under the null hypothesis

### Performance Considerations

- Statistical tests scale linearly with the number of tests
- Complex estimators (e.g., continuous methods) may be computationally expensive
- Consider parallel processing for large-scale analyses
- Memory usage scales with the number of test values stored

## Related Topics

- {ref}`estimator_usage`: General usage patterns for infomeasure estimators
- {ref}`estimator_selection_guide`: Choosing appropriate estimators for your data
- {ref}`settings`: Global configuration options
- {ref}`mutual_information_overview`: Mutual information measures and estimators
- {ref}`transfer_entropy_overview`: Transfer entropy measures and estimators
