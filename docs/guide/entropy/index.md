(entropy_overview)=
# Entropy / Uncertainty /Information

Entropy is the amount of uncertainty associated to a Random variable (RV), on the flip side, this uncertainty is nothing but the lack of information.
The larger the information required to accurately predict the state of RV, the higher is the uncertainty we initially had about it; hence information and uncertainty can be seen as two sides of the same coin.
Thus, the information/uncertainty $H(X)$ associated with a RV $X$ is a quantification of the possibility of making predictions about the occurrence of the next state beyond chance.

## Information Measures
In this python package we will deal with three different measures of entropy, as follows:
### Shannon Entropy
**Hartley** (1928) devised a formula to compute the amount of information associated with an unknown $x$ of which we know nothing except it belongs to the set having $N$ elements, as $ log_2(N)$ {cite:p}`Hartley1928TransmissionOI`.
It was only during 1948, **Claude Shannon** in his seminal paper "_Mathematical Theory of Communication_" {cite:p}`shannonMathematicalTheoryCommunication1948` developed a mathematical measure defined as entropy to quantify the amount of information $H(X)$ produced by a source variable $X$.

$$
H(X) = -\sum_{x \in X} p(x) \log_b p(x),
$$
where,
- $X$: The set of possible values of the random variable.
- $p(x)$: The probability of the value $x$ occurring.
- $b$: The base of the logarithm.
  - If $b = 2$, the unit of information is "bit".
  - If $b = e$, the unit of information is "nat".

> Note: That the base of the algorithm in entropy formula only changes the value of the entropy by a multiplicative constant, hence using one form to another is only a matter of convenience.

The shannon formulation is generic and of which the Hartly would be a special case where all elements are eqi-probable.
Thus, the Shannon entropy quantifies the average amount of information we expect to gain when observing specific outcomes or equivalentely the average decrease in uncertainty about the possible values of a RV.
Shannon’s motivation to use his mathematical formalism (entropy) was to determine whether or not the data stream can be encoded in such a way that even after it has been sent through the channel noisy enough to corrupt the data during the transmission,
the original data stream can be reconstructed in an error-free way at the receiver end. The interested reader can read his paper {cite:p}`shannonMathematicalTheoryCommunication1948` for his findings but here we would like to state another way to understand the Shannon entropy formulation in terms of messages associated with the RV (source).
It is the measure of amount of information in message expressed in binary digits needed to express the message using the most appropiate way to code to get the shortest sequence.


As successful as Shannon’s information theory has been, with time, it was clear that it is capable of dealing with only a limited class of systems one might hope to address in statistical physics.
There was a growing interest to look for more general form of information applicable to diverse complex systems, such as stock market returns, protein folding, percolation, etc.
The additivity of independent mean information was the most natural axiom to attack, see {cite:p}`khinchin1957mathematical` by A.I. Khinchin on axiomatic derivation of Shannon’s entropy.
Basically the axiom states that: if $P$ and $Q$ are discrete generalized probability distributions of two independent random variables, then $H[P Q] = H[P]+H[Q]$.
On this level two modes of reasoning can be formulated. One may either keep the additivity of independent information but utilize more general definition of means, or keep the usual definition of linear means but generalize the additivity law, each leading us to two generalized class of entropies: **R ́enyi entropy**, **Tsallis Entropy**.

### Renyi $\alpha$-Entropy
Alfréd Rényi (mid-60´s) derived generalized family of one-parameter entropies as an exponentially weighted mean of unexpectedness functional (i.e $-log p$) known as **Renyi $\alpha$-Entropy**, for detail check {cite:p}`renyi1976selected` & {cite:p}`jizbaInformationTheoryGeneralized2004`.
Rényi $\alpha$-Entropy is the most general class of information measure preserving the additivity for independent systems and compatible with Kolmogorov´s probability axioms {cite:p}`kolmogoroff1933`.

Let $\alpha > 0$, $\xi_n$ be a random variable with probability distribution $\mathcal{P} = (p_1, ..., p_n)$ where $\sum_{i=1}^{n} p_i = 1$ and $H_\alpha[\mathcal{P}]$ is such that

$$
H_\alpha[\mathcal{P}] := \frac{1}{1-\alpha} \log_2 \left( \sum_{i=1}^{n} p_i^\alpha \right).
$$
where,
- $\alpha$: A positive real number greater than 0.
- $\xi_n$: A random variable.
- $\mathcal{P}$: A probability distribution, given by $\mathcal{P} = (p_1, ..., p_n)$.
- $p_i$: The probability associated with the $i$-th outcome.
- $H_\alpha[\mathcal{P}]$: Rényi entropy of the probability distribution $\mathcal{P}$.

Its distinguishing property is that the small values of probabilities $p_i$ are emphasized for $\alpha < 1$ and, on the contrary, higher probabilities are emphasized for $\alpha > 1$ and for $\alpha = 1$ Rényi entropy reduces to Shannon Entropy.
Rényi $\alpha$-Entropy class can be in particular interesting for the system where additivity (in Shannon sense) is however not always preserved, especially in nonlinear complexsystems, e.g. when we have to deal with long range forces.

### Tsallis Entropy
**Tsallis Entropy (q-order entropy)** or Havrda and Charvát entropy is another possible generalizations of information measure by modifying the additivity law.
Out of the infinity of possible generalizations the so called q–additivity prescription has been used which on q–calculus to formalize the entire approach in an unified manner, for detail refer to  for more details read {cite:p}`articleTsallis` {cite:p}`Tsallis1999`{cite:p}`Tsallis1998`.
The corresponding measure of information is called Tsallis or non–extensive entropy or q-order entropy.

The Tsallis entropy $S_q$ for a probability distribution $\mathcal{P} = (p_1, ..., p_n)$ is defined as

$$
S_q = \frac{1}{1 - q} \left[ \sum_{k=1}^{n} (p_k)^q - 1 \right],
$$
where,
- $\mathcal{P}$: A probability distribution, given by $\mathcal{P} = (p_1, ..., p_n)$.
- $p_k$: The probability associated with the $k$-th outcome.
- $q$: A real number greater than 0, which is a parameter of the entropy.
- $S_q$: Tsallis entropy of the probability distribution $\mathcal{P}$.

In the $q \to 1$ limit, the Jackson sum (q-additivity) reduces to ordinary summation, and the Tallis entropy reduces to Shannon Entropy.
This class of entropy measure is in particularly useful in the study in connection with long–range correlated systems and with non–equilibrium phenomena.

## Estimation Techniques
> Note:
> - This python package consists of non-parametric estimation techniques for both the discrete and continuous RV.
> - All the estimators are based on Shannon information except explicitly mentioned: Rényi entropy estimator & Tsallis entropy estimator

### Discrete and Continuous RV
Real-life data from experiments or observations are recorded in a wide variety of formats. Nevertheless, one can categorize them into discrete and continuous datasets.
A discrete dataset consists of integer values (e.g., 0 and 1) and can be considered the realization of a discrete random variable (RV). Similarly, a continuous dataset consists of real numbers and can be considered the realization of a continuous RV.
Until now, we have not delved into the subtle differences between discrete and continuous random variables, instead using RVs in general. This distinction leads to discrete Shannon information and differential Shannon information for discrete and continuous RVs, respectively.
A simpler way to comprehend entropy formulation is by replacing the summation sign with the integral when moving from discrete to continuous RVs. One thing is clear: information theory is grounded in probability theory.
The entropy measure is seen as a function of the underlying probability distribution function \( p(x) \): the probability mass function (pmf) for discrete RVs and the probability density function (pdf) for continuous RVs.
This python package provides both the discrete estimation technique (for shannon information) and many continuous estimation techniques as will be described in subsequent sections.

### Parametric and non-parametric techniques
When estimating the entropy measure and for that matter the underlying probability distribution function, the approach depends on the system of interest.
Parametric estimation techniques assume the probability distribution falls within a defined family (Gaussian, Poisson, Student-t, etc.) with a shape adjusted by certain parameters.
On the flip side, non-parametric estimation doesn’t commit to any specific distribution shape. This is often the case for systems of interest where there is no prior knowledge of the probability distribution, and the shape may not fit existing families of distributions.
This Python package focuses on non-parametric estimation techniques.


### Bias and Errors
The act of estimating entropic measures associated with real-life datasets always involves **bias** and **variance**. **Bias** is the expected difference between the true value and the estimated value, while **variance** refers to the variability in the estimated value. Thus, every estimation must address errors arising from both bias and variance and strive to minimize their effects.

Minimizing estimation error has led to various estimation techniques, sometimes reporting values in terms of p-values under certain null hypotheses. The diversity in estimation techniques arises from factors such as computational cost, the nature of dataset availability, the specific question at stake, and so on. Therefore, users must be diligent in selecting appropriate estimators.

### List of Estimation Techniques Implemented:

```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Discrete RV

   discrete

.. toctree::
   :maxdepth: 2
   :caption: Continuous RV

   kernel
   kozachenko_leonenko
   symbolic
   renyi
   tsallis
