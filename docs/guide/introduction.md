# Introduction
In this era of modernity, the systems we study and the problems we tackle are becoming increasingly complex, demanding innovative approaches to address them.
One such approach involves leveraging **Information Theory** {cite:p}`shannonMathematicalTheoryCommunication1948`.
The core idea is to distill any given problem into its fundamental informational components and analyze the underlying dynamics through the lens of information sharing and transfer.
In recent years, **information-theoretic measures**—such as entropy, mutual information, and transfer entropy—have gained significant traction across diverse scientific disciplines {cite:p}`Lizier2014,versteegInformationtheoreticMeasuresInfluence2013,acharya2024representative`.
Researchers from various fields, many of whom are not formally trained in information theory, often seek to apply these measures to their specific problems of interest.
However, a common challenge arises: despite the growing interest, there is often a lack of accessible tools that allow users to estimate these measures using their preferred estimation techniques.
This Python package is designed for anyone looking to implement **information-theoretic measures** within their field of study.
It provides comprehensive descriptions and implementations of these measures, making them accessible and practical to use.

This package includes key measures in information theory,
as developed by the principles of Shannon:
- Entropy (H)
- Conditional Entropy (CH)
- Cross-Entropy (CE)
- Mutual Information (MI)
- Conditional Mutual Information (CMI)
- Transfer Entropy (TE)
- Conditional Transfer Entropy (CTE)
- Jensen-Shannon Divergence (JSD)
- Kulback-Leibler Divergence (KLD)

Concerning entropy generalizations, we have Rényi and Tsallis entropy, and the further measures that arise from them.

# Estimation
Experimental or observational data come in various formats but generally fall into discrete or continuous categories.
Discrete datasets consist of integer values (integers or categorical variables, e.g., in ℤ) and represented as the realization of discrete random variables (RVs).
Continuous datasets contain real numbers (ℝ) and can be represented as realization of continuous RVs.
The **probability mass function (pmf)** defines discrete RVs while the **probability density function (pdf)** applies to continuous RVs.

```{note}
This package provides estimation techniques for both discrete and continuous variables.
```

When estimating information theoretic measures—especially the underlying probability distribution function $ p(x)$—one must choose between **parametric** and **non-parametric** techniques to begin with.
- **Parametric estimation** assumes $ p(x)$ belongs to a known family (e.g., Gaussian, Poisson, Student-t), with its shape defined by a set of parameters.
- **Non-parametric estimation** makes no such assumptions, making it ideal when the distribution is unknown or doesn’t fit standard families.

```{note}
This package focuses on non-parametric estimation techniques.
```

Estimating information measures—and indeed any other type of measure—from real-life data inherently involves
two key issues: **bias**, which is the expected difference between true values and estimated values, and
**variance**, which refers to the variability or spread in the estimates.
To ensure accuracy, estimation techniques must minimize both.
This package offers a variety of estimation methods, allowing users to choose the most suitable one.
Additionally, it provides an option to compute **p-values** for measures like Mutual Information (MI) and Transfer Entropy (TE) by assuming no relationship as the null hypothesis.
The corresponding **t-scores** are also provided.
For TE, we implement effective Transfer Entropy (eTE), a method designed to reduce bias from finite sample effects.

```{admonition} This Package
- allows users to compute **p-values** for MI and TE to assess significance.
- includes  **effective Transfer Entropy (eTE)**, reducing bias from finite sample sizes.
```

Furthermore, **local values** can be computed, providing insights into the dynamic of the system being studied {cite:p}`Lizier2014,fano1961transmission,local_TE_Lizier`.

## Types of Estimation techniques available
For discrete variables, Shannon's initial, discrete estimation techniques are used. For continuous variables, three methods have been implemented:
- Kernel Estimation
- Ordinal / Symbolic / Permutation Estimation
- Kozachenko-Leonenko (KL) / Kraskov-Stoegbauer-Grassberger (KSG) / Metric / kNN Estimation

Let's compile all the estimation techniques along with the corresponding Shannon information measures they can estimate into a single table, as shown below:
:::{list-table} Information Measures and Estimation Methods
:name: information-measures
:widths: 2 1 1 1 1 1
:header-rows: 1
:stub-columns: 1

*   - Measures \ Estimators
    - Notation
    - Discrete Estimator
    - Kernel Estimator
    - Metric / kNN Estimator
    - Ordinal Estimator
*   - {ref}`Entropy <entropy_overview>`
    - $H(X)$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Rényi <renyi_entropy>` & {ref}`Tsallis <tsallis_entropy>` entropies
    - $H(X)$
    -
    -
    - ✓
    -
*   - {ref}`Joint Entropy`[^renyi_tsallis]
    - $H(X,Y)$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Cross-Entropy <cross_entropy_overview>`[^renyi_tsallis]
    - $H_Q(P)$[^cross-nomenclature]
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Mutual Information (MI) <mutual_information_overview>`[^renyi_tsallis]
    - $I(X;Y)$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Conditional MI <cond_mi_overview>`[^renyi_tsallis]
    - $I(X;Y|Z)$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Transfer Entropy (TE) <transfer_entropy_overview>`[^renyi_tsallis]
    - $T_{X \to Y}$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Conditional TE <cond_te_overview>`[^renyi_tsallis]
    - $T_{X \to Y|Z}$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Kullback-Leibler Divergence (KLD) <kullback_leibler_divergence>`
    - $\operatorname{KLD}(P||Q)$
    - ✓
    - ✓
    - ✓
    - ✓
*   - {ref}`Jensen Shannon Divergence (JSD) <jensen_shannon_divergence>`
    - $\operatorname{JSD}(P||Q)$
    - ✓
    - ✓
    -
    - ✓
:::

For Rényi and Tsallis, MI, CMI, TE and CTE use entropy combination formulas internally, as well as the composite measures JSD and KLD.
In all other cases, this package uses probabilistic formulas, as these introduce less bias.

[^cross-nomenclature]: Nomenclature taken from Christopher Olah's blog post [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/#fnref4).
We choose this nomenclature $H_Q(P)$; as the widely used $H(p, q)$ would be ambiguous with joint entropy.

[^renyi_tsallis]: These measures can also be computed using the Rényi and Tsallis
entropy formulations, in addition to the standard Shannon entropy formulation.
