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
     <a href="https://arxiv.org/abs/2505.14696" style="margin: 0 10px; display: inline-block;">
       <img src="https://img.shields.io/badge/arXiv-2505.14696-b31b1b.svg" alt="arXiv Pre-print" />
     </a>
     <a href="https://doi.org/10.5281/zenodo.15241810" style="margin: 0 10px; display: inline-block;">
       <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15241810.svg" alt="Zenodo Project" />
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
   <div style="height: 20px;"></div>

```

::::{grid} 1 2 2 3
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

:::{grid-item-card} {material-regular}`lightbulb;2em` Demos
:link: demos
:link-type: ref

A collection of short demos showcasing the capabilities of this package.\
E.g., analytical comparison and paper reproduction.
+++
{ref}`Learn more »<Demos>`
:::

::::

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

If you use `infomeasure` in your research, please cite our [pre-print](https://arxiv.org/abs/2505.14696) (submitted).
You can also find citation information for this project in the `CITATION.cff` file in [the repository](https://github.com/cbueth/infomeasure) and cite it accordingly.
Alternatively, if you'd like to cite the software itself or a specific version, find the [Zenodo project page](https://doi.org/10.5281/zenodo.15241810)
for the specific version you are using and cite it accordingly.

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
