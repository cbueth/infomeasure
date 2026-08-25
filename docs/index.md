---
sd_hide_title: true
site:
  options:
    hide_toc: true
---
(infomeasure_docs)=
# Overview

:::{image} _static/im_logo_transparent.png
  :width: 700
  :align: center
  :class: only-light
  :alt: infomeasure logo
  :target: .
:::

:::{image} _static/im_logo_transparent_dark.png
  :width: 700
  :align: center
  :class: only-dark
  :alt: infomeasure logo
  :target: .
:::


```{eval-rst}
.. raw:: html

   <div style="height: 10px;"></div>
   <div style="text-align: center;">
     <a href="https://pypi.org/project/infomeasure/" style="margin: 0 10px; display: inline-block;">
       <img src="https://badge.fury.io/py/infomeasure.svg" alt="PyPI version" />
     </a>
     <a href="https://anaconda.org/conda-forge/infomeasure" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/conda/vn/conda-forge/infomeasure.svg" alt="Conda version" />
     </a>
     <a href="https://pypi.org/project/infomeasure/" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/pypi/pyversions/infomeasure" alt="Python version" />
     </a>
     <a href="https://pypi.org/project/infomeasure/" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/pypi/l/infomeasure" alt="License" />
     </a>
   </div>
   <div style="height: 5px;"></div>
   <div style="text-align: center;">
     <a href="https://doi.org/10.1038/s41598-025-14053-5" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/static/v1?label=Sci.%20Rep.&message=10.1038/s41598-025-14053-5&color=005b96" alt="Scientific Reports" />
     </a>
     <a href="https://arxiv.org/abs/2505.14696" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/badge/arXiv-2505.14696-b31b1b.svg" alt="arXiv Pre-print" />
     </a>
     <a href="https://doi.org/10.5281/zenodo.15241810" style="margin: 0 10px; display: inline-block;">
       <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15241810.svg" alt="Zenodo Project" />
     </a>
     <a href="https://www.semanticscholar.org/paper/eab2671b69dae18ed1075195d700bbb98dc4712d" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.openalex.org%2Fworks%2FW4413108477%3Fselect%3Dcited_by_count&query=%24.cited_by_count&label=cited%20by&suffix=%2B&color=005b96" alt="Cited by" />
     </a>
   </div>
   <div style="height: 20px;"></div>

```

::::{grid} 1 2 2 4
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`rocket;2em` Getting Started
:link: getting_started
:link-type: ref

How to install this package and run the first calculation.\
Start your endeavour here!

+++
{ref}`Learn more » <getting_started>`
:::

:::{grid-item-card} {material-regular}`menu_book;2em` Reference Guide
:link: reference_guide
:link-type: ref

Theoretic background of the library.
See all estimation techniques with code snippets.

+++
{ref}`Learn more »<reference_guide>`
:::

:::{grid-item-card} {material-regular}`psychology;2em` Estimator Selection
:link: estimator_selection_guide
:link-type: ref

Discover which estimator works best for your data with our interactive decision tree.

+++
{ref}`Learn more » <estimator_selection_guide>`
:::

:::{grid-item-card} {material-regular}`lightbulb;2em` Demos
:link: demos
:link-type: ref

Short demos showcasing package capabilities and analytical comparisons.
+++
{ref}`Learn more »<Demos>`
:::

::::

```{admonition} Rust implementation now available!
:class: important
All core measures have been reimplemented in
[`infomeasure-rs`](https://crates.io/crates/infomeasure)
([crate docs](https://docs.rs/infomeasure)) with
compile-time type safety, GPU acceleration, and significantly faster execution.
Check out the [Rust Guide](https://docs.rs/infomeasure/latest/infomeasure/guide/index.html)
if you need maximum performance for production or large-scale analysis.
```

## What is `infomeasure`?

`infomeasure` is a Python library for computing information measures, such as entropy,
mutual information and conditional mutual information.
It provides a simple and efficient way to compute these measures on large datasets.
The {ref}`Reference pages <reference_guide>` provide a comprehensive, theoretical background on the concepts behind these measures, while the {ref}`Demos` provide practical examples of how to use `infomeasure` in real-world applications.

## Setup and use

To set up `infomeasure`, see the {ref}`Getting Started` page, more on
the details of the inner workings can be found on the {ref}`Reference pages <reference_guide>`.
Furthermore, you can also find the {ref}`API documentation <API Reference>`.
The introduction talk has been recorded and can be seen on
the [IFISC YouTube channel](https://www.youtube.com/watch?v=ckScv1E-vHE) and the
[slides here](https://carlson.pages.ifisc.uib-csic.es/infomeasure-introduction-presentation/lab/index.html?path=infomeasure-presentation.ipynb).

## How to cite

If you use `infomeasure` in your research, please cite our paper
in [Scientific Reports](https://doi.org/10.1038/s41598-025-14053-5).
You can also find citation information for this project in the `CITATION.cff` file
in [the repository](https://github.com/cbueth/infomeasure) and cite it accordingly.
Alternatively, if you'd like to cite the software itself or a specific version,
find the [Zenodo project page](https://doi.org/10.5281/zenodo.15241810)
for the specific version you are using and cite it accordingly.

## Cited in research

`infomeasure` is used across research fields *(selection, as of August 2026)*:

- A. Berke, E. Bacis and U. Syed,
  [An Improved Entropy Measure for Web Browser Fingerprinting Risk](https://doi.org/10.56553/popets-2026-0102),
  _Proceedings on Privacy Enhancing Technologies_, 2026.
- Y. Bel-Hadj et al.,
  [Inferring wind turbine operational state and fatigue from high-frequency acceleration using self-supervised learning for SCADA-free monitoring](https://doi.org/10.5194/wes-11-1363-2026),
  _Wind Energy Science_, 2026.
- L. Tiawongsuwan et al.,
  [Autism spectrum disorder disrupts brain network connectivity maturation during childhood development](https://doi.org/10.1038/s41598-025-30971-w),
  _Scientific Reports_, 2025.
- L. H. McCabe and H. H. Huang,
  [SENECA: Small-Sample Discrete Entropy Estimation via Self-Consistent Missing Mass](https://arxiv.org/abs/2605.00668),
  _arXiv:2605.00668_, 2026.
- R. García-Leal et al.,
  [Functional Connectivity Between Human Motor and Somatosensory Areas During a Multifinger Tapping Task: A Proof-of-Concept Study](https://www.mdpi.com/2673-4087/7/1/12),
  _NeuroSci_, 2026.

See all citations on
[Semantic Scholar](https://www.semanticscholar.org/paper/eab2671b69dae18ed1075195d700bbb98dc4712d).

## Contributing
If you want to contribute to the development of `infomeasure`, please read the
[CONTRIBUTING.md](https://github.com/cbueth/infomeasure/blob/main/CONTRIBUTING.md)
file.

## Acknowledgments

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 851255).
This work was partially supported by the María de Maeztu project CEX2021-001164-M funded by the MICIU/AEI/10.13039/501100011033 and FEDER, EU.


```{eval-rst}
.. toctree::
   :hidden:
   :name: table_of_contents
   :caption: Table of Contents
   :maxdepth: 1
   :glob:

   getting_started
   guide/index
   demos/index
   api/index
   changelog
   bibliography
```
