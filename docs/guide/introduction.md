# Introduction
In this era of modernity, the systems we study and the problems we tackle are becoming increasingly complex, demanding innovative approaches to address them.
One such approach involves leveraging **Information Theory** (cite).
The core idea is to distill any given problem into its fundamental informational components and analyze the underlying dynamics through the lens of information sharing and transfer.
In recent years, **information-theoretic measures**—such as entropy, mutual information, and transfer entropy—have gained significant traction across diverse scientific disciplines (cite).
Researchers from various fields, many of whom are not formally trained in information theory, often seek to apply these measures to their specific problems of interest.
However, a common challenge arises: despite the growing interest, there is often a lack of accessible tools that allow users to estimate these measures using their preferred estimation techniques.
This Python package is designed for anyone looking to implement **information-theoretic measures** within their field of study.
It provides comprehensive descriptions and implementations of these measures, making them accessible and practical to use.

This package includes key measures in information theory,
as developed by the principles of Shannon:
- Entropy (H)
- Conditional Entropy (CH)
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
- Rényi Estimation
- Tsallis Estimation

Let's compile all the estimation techniques along with the corresponding Shannon information measures they can estimate into a single table, as shown below:
:::{list-table} Information Measures and Estimation Methods
:name: information-measures
:widths: 2 1 1 1 1 1 1
:header-rows: 1
:stub-columns: 1

*   - Measures
    - Notation
    - Discrete Estimator
    - Kernel Estimator
    - KL/KSG Estimator
    - Ordinal Estimator
    - Rényi & Tsallis Estimator
*   - Entropy
    - $H(X)$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
*   - Joint Entropy
    - $H(X,Y)$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
*   - Mutual Information (MI)
    - $I(X;Y)$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
*   - Conditional MI
    - $I(X;Y|Z)$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
*   - Transfer Entropy (TE)
    - $T_{X \to Y}$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
*   - Conditional TE
    - $T_{X \to Y|Z}$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
*   - Jensen Shannon Divergence (JSD)
    - $\operatorname{JSD}(P||Q)$
    - ✓
    - ✓
    -
    - ✓
    - 
*   - Kullback-Leibler Divergence (KLD)
    - $\operatorname{KLD}(P||Q)$
    - ✓
    - ✓
    - ✓
    - ✓
    - ✓
:::

For Rényi and Tsallis, MI, CMI, TE and CTE use entropy combination formulas internally, as well as the composite measures JSD and KLD.
In all other cases, this package uses probabilistic formulas, as these introduce less bias.