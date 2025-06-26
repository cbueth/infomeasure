# Changelog

## Unreleased

This release introduces an overhaul of the statistical testing functionality with breaking changes to the API.

- 🚨 **BREAKING CHANGES**:

  - **Removed** `p_value()` and `t_score()` methods from `PValueMixin`.

  - **Replaced** with comprehensive {func}`~infomeasure.estimators.base.PValueMixin.statistical_test` method that returns a {class}`~infomeasure.estimators.base.StatisticalTestResult` object.

  - **Renamed** configuration parameter `p_value_method` to `statistical_test_method`.

  - **Added** new configuration parameter `statistical_test_n_tests` for default number of tests, see {ref}`Config`.

- ✨ **New Features**:

  - 📊 **Comprehensive Statistical Testing**: New {func}`~infomeasure.estimators.base.PValueMixin.statistical_test` method provides _p_-values, _t_-scores, and metadata in a single call.

  - 📈 **StatisticalTestResult Class**: Rich result object containing:

    - _p_-value and _t_-score
    - Test values from resampling
    - Observed value and null distribution statistics
    - Number of tests and method used

  - 📊 **Flexible Percentile Access**: {func}`~infomeasure.estimators.base.StatisticalTestResult.percentile` method wraps numpy's percentile function for test values.

  - 🎯 **Convenience Confidence Intervals**: {func}`~infomeasure.estimators.base.StatisticalTestResult.confidence_interval` method for easy CI calculation.

- 🔧 **API Improvements**:

  - **Simplified Interface**: No need to specify confidence levels upfront - calculate on demand.

  - **Better Metadata**: Statistical results include test method and number of tests used.

  - **Consistent Return Types**: All statistical operations return structured objects.

- 🧮 **Added Estimators**:

  - **Miller-Madow Estimators**: Comprehensive suite of bias-corrected information measure estimators using the Miller-Madow correction formula. Provides improved estimates for small sample sizes by adding correction terms to maximum likelihood estimates.

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

    - **Chao-Shen Entropy**: {class}`~infomeasure.estimators.entropy.chao_shen.ChaoShenEntropyEstimator` - Bias-corrected entropy estimator that accounts for unobserved species through coverage estimation using singleton counts, providing improved estimates for incomplete sampling scenarios.

    - **Shrinkage Entropy**: {class}`~infomeasure.estimators.entropy.shrink.ShrinkEntropyEstimator` - James-Stein shrinkage entropy estimator that applies shrinkage to probability estimates before computing entropy, reducing bias in small sample scenarios through regularization toward uniform distribution.

    - **Grassberger Entropy**: {class}`~infomeasure.estimators.entropy.grassberger.GrassbergerEntropyEstimator` - Discrete entropy estimator with finite sample corrections using the digamma function, providing bias-corrected entropy estimates through count-based corrections.

    All new entropy estimators include comprehensive test coverage and support for local values calculation where applicable.

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

- ✨ Optimizations and bug fixes:

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

- 💡 Enhanced performance through optimizations in `base.py`.

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
