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
   <div style="height: 20px;"></div>

.. hidden:: to be added when doi available
     <a href="https://zenodo.org/badge/latestdoi/{TBA}" style="margin: 0 10px; display: inline-block;">
       <img src="https://zenodo.org/badge/{TBA}.svg" alt="Zenodo Project" />
     </a>
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
An analytical comparison, model and data application.

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

## How to cite

If you use `infomeasure` in your research, find the `CITATION.cff` file in [the repository](https://github.com/cbueth/infomeasure) and cite it accordingly.
GitHub provides a "Cite this repository" button on the right side of the page for an APA and BibTeX citation.

## Contributing
If you want to contribute to the development of `infomeasure`, please read the
[CONTRIBUTING.md](https://github.com/cbueth/infomeasure/blob/main/CONTRIBUTING.md)
file.


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
