# Changelog

## Under Development

- 📈 **Variational MI estimators**: For large datasets, stochastic variational inference {cite:p}`hoffmanStochasticVariationalInference2013` becomes a valid approach to determine variational bounds of mutual information (MI). The following variational estimators for MI are planned:

  - **DV** {cite:p}`donskerAsymptoticEvaluationCertain1975`: The Donsker-Varadhan estimator provides a dual formulation of the KL-divergence for variational MI bounds, forming the theoretical foundation for many neural MI estimators.

  - **BA** {cite:p}`barberIMAlgorithmVariational2003`: The Barber-Agakov estimator uses variational approximation to compute MI over noisy channels, similar to the EM algorithm but maximizing MI instead of a likelihood. It introduces a tractable lower bound by replacing intractable conditional distributions with variational approximations.

  - **MINE** {cite:p}`belghaziMutualInformationNeural2018`: Mutual Information Neural Estimation employs gradient descent over neural networks to estimate MI between high-dimensional continuous variables. It is scalable in dimensionality and sample size, trainable through back-propagation, and strongly consistent.

  - **NWJ** {cite:p}`nguyenEstimatingDivergenceFunctionals2010`: The Nguyen-Wainwright-Jordan estimator uses convex risk minimization to estimate divergence functionals and likelihood ratios through f-divergence characterization. This approach leverages convexity to ensure robust and efficient estimation. Furthermore, a CMI bound can be obtained using {cite:p}`molavipourConditionalMutualInformation2020`, enabling estimation of TE and CTE.

  - **JSD** {cite:p}`hjelmLearningDeepRepresentations2018`: This estimator uses Jensen-Shannon divergence for representation learning by maximizing MI between input and encoder output, incorporating locality structure and adversarial matching for unsupervised learning.

  - **TUBA** {cite:p}`pooleVariationalBoundsMutual2019`: Tractable Unnormalized Barber and Agakov estimator provides unbiased estimates and gradients using energy-based variational families to avoid intractable partition functions while maintaining tractability.

  - **NCE** {cite:p}`oordRepresentationLearningContrastive2019,maNoiseContrastiveEstimation2018`: A multi-sample mutual information estimator based on noise contrastive estimation (NCE) {cite:p}`gutmannNoisecontrastiveEstimationUnnormalized2012`.

  - $\bf{I_{\alpha}}$ {cite:p}`pooleVariationalBoundsMutual2019`: This interpolated bound balances variance and square bias, providing a flexible trade-off between bias and variance controlled by the parameter α.

  - **FLO** {cite:p}`guoTightMutualInformation2022`: Fenchel-Legendre Optimization offers a novel contrastive MI estimator that overcomes InfoNCE limitations by achieving tight bounds and provable convergence. It uses unnormalized statistical modeling and convex optimization to improve data efficiency.

- **EEVI**: Estimators of entropy via inference, i.e. using sequential Monte Carlo {cite:p}`saadEstimatorsEntropyInformation2022`.

- **Automatic evaluation of multiple time lags**: For MI and TE estimators, automatic evaluation of multiple time lags to find optimal lag parameters and improve information measure accuracy.



## Version 0.6.0 (unreleased)

This release implements the KSG Type I and Type II variants and improves neighbor counting robustness.

- 🚨 **BREAKING CHANGES**:

  - **Neighbor Counting Logic**: The default behavior for KSG-based estimators (MI, CMI, TE, CTE) has been updated to strictly follow Type I logic (strict inequality `dist < eps` for marginal counts). This may lead to different results compared to previous versions, especially for data with many identical values.

- ✨ **New Features**:

  - **KSG Variants**: Added `ksg_id` parameter (1 or 2) to all KSG-based estimators.
    - `ksg_id=1` (Type I): Original KSG estimator with strict inequality for marginal counts.
    - `ksg_id=2` (Type II): KSG variant with non-strict inequality (`dist <= eps`) and modified formula.
  - **Joint Conditional Variables**: Discrete CMI and CTE estimators now support multidimensional conditioning variables (joint variables) by automatically reducing them to a single joint space.
  - **KNN Entropy Improvements**: `KozachenkoLeonenkoEntropyEstimator` now also supports `ksg_id` to switch between standard and modified formulas.
  - **Standardized Metrics**: Ensured consistent use of the Minkowski metric across joint and marginal spaces in all KNN-based estimators.
  - **Vectorized Counting**: Improved performance of neighbor counting using SciPy's vectorized `query_ball_point`.

## Version 0.5.1 (2026-01-17)

Added support for Python 3.14.

## Version 0.5.0 (2025-07-02)

This release introduces an overhaul of the statistical testing functionality with breaking changes to the API.

- 🚨 **BREAKING CHANGES**:

  - **Removed** `p_value()` and `t_score()` methods from `PValueMixin`.

  - **Replaced** with comprehensive {func}`~infomeasure.estimators.mixins.StatisticalTestingMixin.statistical_test` method that returns a {class}`~infomeasure.utils.data.StatisticalTestResult` object.

  - **Renamed** configuration parameter `p_value_method` to `statistical_test_method`.

  - **Added** new configuration parameter `statistical_test_n_tests` for default number of tests, see {ref}`Config`.

- ✨ **New Features**:

  - 📊 **Comprehensive Statistical Testing**: New {func}`~infomeasure.estimators.mixins.StatisticalTestingMixin.statistical_test` method provides _p_-values, _t_-scores, and metadata in a single call.

  - 📈 **StatisticalTestResult Class**: Rich result object containing:

    - _p_-value and _t_-score
    - Test values from resampling
    - Observed value and null distribution statistics
    - Number of tests and method used

  - 📊 **Flexible Percentile Access**: {func}`~infomeasure.utils.data.StatisticalTestResult.percentile` method wraps numpy's percentile function for test values.

  - 🎯 **Convenience Confidence Intervals**: {func}`~infomeasure.utils.data.StatisticalTestResult.confidence_interval` method for easy CI calculation.

- 🔧 **API Improvements**:

  - **Simplified Interface**: No need to specify confidence levels upfront - calculate on demand.

  - **Better Metadata**: Statistical results include test method and number of tests used.

  - **Consistent Return Types**: All statistical operations return structured objects.

- 🧮 **Added Estimators**:

  - **Miller-Madow Estimators**: Comprehensive suite of bias-corrected information measure estimators using the Miller-Madow correction formula. Provides improved estimates for small sample sizes by adding correction terms to maximum likelihood estimates. These estimators are dedicated implementations.

    - **Entropy (H)**: {class}`~infomeasure.estimators.entropy.miller_madow.MillerMadowEntropyEstimator` with correction term `(K-1)/(2N)` for bias-corrected entropy estimation.

    - **Mutual Information (MI)**: {class}`~infomeasure.estimators.mutual_information.miller_madow.MillerMadowMIEstimator` for bias-corrected mutual information with support for arbitrary number of variables.

    - **Conditional Mutual Information (CMI)**: {class}`~infomeasure.estimators.mutual_information.miller_madow.MillerMadowCMIEstimator` for bias-corrected conditional mutual information.

    - **Transfer Entropy (TE)**: {class}`~infomeasure.estimators.transfer_entropy.miller_madow.MillerMadowTEEstimator` for bias-corrected transfer entropy with statistical testing support.

    - **Conditional Transfer Entropy (CTE)**: {class}`~infomeasure.estimators.transfer_entropy.miller_madow.MillerMadowCTEEstimator` for bias-corrected conditional transfer entropy.

    - **Kullback-Leibler Divergence (KLD)**: Miller-Madow correction available through `approach="millermadow"` or `approach="mm"` in {func}`~infomeasure.kld`.

    - **Jensen-Shannon Divergence (JSD)**: Miller-Madow correction available through `approach="millermadow"` or `approach="mm"` in {func}`~infomeasure.jsd`.

    All Miller-Madow estimators include comprehensive test coverage and support for local values calculation where applicable.

  - **Additional Entropy Estimators**: New discrete entropy estimators with specialized bias correction and estimation techniques:

    - **Bayesian Entropy**: {class}`~infomeasure.estimators.entropy.bayes.BayesEntropyEstimator` - Bayesian entropy estimator with concentration parameter α supporting multiple prior choices (Jeffrey, Laplace, Schurmann-Grassberger, Minimax) for improved entropy estimation with prior knowledge incorporation.

    - **Chao-Shen Entropy**: {class}`~infomeasure.estimators.entropy.chao_shen.ChaoShenEntropyEstimator` - Bias-corrected entropy estimator that accounts for unobserved species through coverage estimation using singleton counts, providing improved estimates for incomplete sampling scenarios {cite:p}`chaoNonparametricEstimationShannons2003`.

    - **Shrinkage Entropy**: {class}`~infomeasure.estimators.entropy.shrink.ShrinkEntropyEstimator` - James-Stein shrinkage entropy estimator that applies shrinkage to probability estimates before computing entropy, reducing bias in small sample scenarios through regularization toward uniform distribution {cite:p}`hausserEntropyInferenceJamesStein2009`.

    - **Grassberger Entropy**: {class}`~infomeasure.estimators.entropy.grassberger.GrassbergerEntropyEstimator` - Discrete entropy estimator with finite sample corrections using the digamma function, providing bias-corrected entropy estimates through count-based corrections {cite:p}`grassbergerFiniteSampleCorrections1988,grassbergerEntropyEstimatesInsufficient2008`.

    - **Chao Wang Jost Entropy**: {class}`~infomeasure.estimators.entropy.chao_wang_jost.ChaoWangJostEntropyEstimator` - Advanced bias-corrected entropy estimator that uses coverage estimation based on singleton and doubleton counts to account for unobserved species, providing improved entropy estimates for incomplete sampling scenarios with sophisticated statistical corrections {cite:p}`chaoEntropySpeciesAccumulation2013`.

    - **ANSB Entropy**: {class}`~infomeasure.estimators.entropy.ansb.AnsbEntropyEstimator` - Asymptotic NSB entropy estimator designed for extremely undersampled discrete data where the number of unique values K is comparable to the sample size N. Uses the formula (γ - log(2)) + 2 log(N) - ψ(Δ) where γ is Euler's constant, ψ is the digamma function, and Δ is the number of coincidences, providing efficient entropy estimation in the undersampled regime {cite:p}`nemenmanEntropyInformationNeural2004`.

    - **NSB Entropy**: {class}`~infomeasure.estimators.entropy.nsb.NsbEntropyEstimator` - Nemenman-Shafee-Bialek entropy estimator providing Bayesian estimates of Shannon entropy for discrete data using numerical integration. Particularly effective for undersampled data where traditional estimators may be biased, using a principled Bayesian approach that accounts for sampling uncertainty through integration over possible entropy values {cite:p}`nemenmanEntropyInferenceRevisited2002`.

    - **Zhang Entropy**: {class}`~infomeasure.estimators.entropy.zhang.ZhangEntropyEstimator` - Zhang entropy estimator for discrete data using the recommended definition from Grabchak et al. {cite:p}`grabchakAuthorshipAttributionUsing2013`. Implements the fast calculation approach from Lozano et al. {cite:p}`lozanoFastCalculationEntropy2017` with bias correction through sophisticated probability weighting. Provides improved entropy estimates through advanced statistical corrections while maintaining computational efficiency.

    - **Bonachela Entropy**: {class}`~infomeasure.estimators.entropy.bonachela.BonachelaEntropyEstimator` - Bonachela entropy estimator designed for small data sets using the formula from Bonachela et al. {cite:p}`bonachelaEntropyEstimatesSmall2008`. Provides a compromise between low bias and small statistical errors for short data series, particularly effective when data sets are small and probabilities are not close to zero.

    These new estimators were selected based on {cite:p}`degregorioEntropyEstimatorsMarkovian2024`
    and the implementations of [DiscreteEntropy.Jl](https://github.com/kellino/DiscreteEntropy.jl)
    were consulted for help {cite:p}`kellyDiscreteEntropyjlEntropyEstimation2024`.

  - **Complete Estimator Coverage**: The new estimators also all support MI, CMI, TE and CTE, using the same unified slicing, integrated into the interface.

  - **Jensen-Shannon Divergence Support**: Of the new estimators {func}`~infomeasure.composite_measures.jsd.jensen_shannon_divergence` is available for {class}`~infomeasure.estimators.entropy.bayes.BayesEntropyEstimator` and {class}`~infomeasure.estimators.entropy.shrink.ShrinkEntropyEstimator`.

  - **Enhanced Cross-Entropy and KLD Support**: Of the new estimators, Bayes and Miller-Madow support cross entropy and thus also the {func}`~infomeasure.composite_measures.kld.kullback_leiber_divergence`. All entropies which have been implemented before version `0.5.0` all support cross entropy and KLD already.

- 📚 Update Documentation

- 🧪 Updated tests

---

## Version 0.4.0 (2025-05-02)

The `0.4.0` release introduces cross-entropy support, improves code packaging, and enhances documentation.

- 📈 **Cross-Entropy support**:

  - Added cross-entropy for all approaches.

  - Integrated cross-entropy into the documentation with detailed explanations and examples.

  - Restricted the use of joint random variables (RVs) for cross-entropy to avoid ambiguity.

- 📦 **Code packaging**:

  - 📦 Added tests to packaged tarball for testing in `conda-forge`.

  - 🔧 Updated deprecated licence classifier.

  - 🔧 Added Zenodo integration and updated README.md with logo and badges.

  - 🔧 Added README.md formatting for logos and badges.

- 🔧 **Warnings handling**: Handled warnings as errors in pytest and addressed warnings in the code.

- 📚 **Documentation**:

  - 📚 Added a benchmark demo page to documentation.

  - 📄 Added acknowledgments and funding information.

  - 🎨 Updated logo and icon design.

  - 🔧 Added favicon and polished documentation index page, including logo and dark mode support.

  - 🔧 Added demos for Gaussian data and Schreiber Article.

  - 📊 Changed Gaussian axis titles and corrected Schreiber Demo information unit.

  - 🔧 Changed links and reformatted documentation.

---

## Version 0.3.3 (2025-04-16)

The `0.3.3` release focuses on improving documentation, moving to Read the Docs, and polishing the project.

- 📚 Improved documentation and moved to [Read the Docs](https://infomeasure.readthedocs.io/).

  - 📄 Added `automodapi` for estimators and `sphinx-apidoc`.

  - 📊 Added `graphviz` apt dependency and fixed requirement structure.

  - 📝 Added code examples and reworked guide pages.

  - 🔗 Changed URL and repository settings.

- 📦 Updated project for publication.

- ✨ Optimisations and bug fixes:

  - 🚀 Parallelized box and Gaussian kernel calculations.

  - 🔄 Reused parameters between p-value and t-score calculations.

  - 🔧 Fixed bootstrap resampling for inhomogeneous, higher-dimensional input data.

  - 🔧 Optimized kernel (C)TE calculations.

  - 🔧 Fixed calling t-score without p-value.

---

## Version 0.3.0 (2025-04-01)

The `0.3.0dev0` release focuses on performance improvements, feature enhancements, and API updates.

- 🔧 **Local values support**: All approaches now support local values.

- 🎯 Added two new composite measures:

  - Jensen-Shannon Divergence (JSD)

  - Kullback-Leibler Divergence (KLD)

- ✨ Optimized algorithms for:

  - Mutual Information (MI) and Conditional Mutual Information (CMI) on discrete and ordinal data.

  - Transfer Entropy (TE) and Conditional Transfer Entropy (CTE).

- ⚡ Major API refactoring to improve compatibility with arbitrary many random variables in MI and CMI.

- 💡 Enhanced performance through optimisations in `base.py`.

- 🔍 Added extensive testing for local values and tested manually with code notebooks.

- ⬆️ Added Python 3.13 support.

---

## Version 0.2.1 (2025-02-11)

The `0.2.1dev0` release marks the first release, providing essential information
measures and estimators like Entropy (H), Mutual Information (MI), and others.
It includes a CI/CD pipeline, supports Python 3.10-3.12, and is licensed under AGPLv3+.

- 📦 **First release** of the `infomeasure` package.

- 🧩 Added essential information measure estimators:

  - Shannon entropy (H)
  - Mutual Information (MI)
  - Conditional Mutual Information (CMI)
  - Transfer Entropy (TE) and Conditional Transfer Entropy (CTE)
  - Jensen-Shannon Divergence (JSD)
  - Kullback-Leibler Divergence (KLD)

- 🔄 Set up CI/CD pipeline with GitLLab CI.

- 💻 Added support for Python 3.10+.

- 📄 Updated documentation to include installation guide, package structure,
  and example use cases.

---

## Version 0.0.0 (2024-06-06)

- Package setup

  - 🏗 Written `pyproject.toml`
  - 🔄 General project and test structure with CI/CD
  - 📚️ Documentation with `sphinx`, `sphinxcontrib-bibtex` and `numpydoc`
