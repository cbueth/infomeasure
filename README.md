<div style="text-align: center; max-width: 700px; margin: 0 auto;">
  <a href="https://infomeasure.readthedocs.io/">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/cbueth/infomeasure/refs/heads/main/docs/_static/im_logo_transparent.png">
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cbueth/infomeasure/refs/heads/main/docs/_static/im_logo_transparent_dark.png">
      <img src="https://raw.githubusercontent.com/cbueth/infomeasure/refs/heads/main/docs/_static/im_logo_transparent.png" style="max-width: 100%; height: auto;" alt="infomeasure logo">
    </picture>
  </a>
</div>

<div align="center">

<a href="">[![Documentation](https://readthedocs.org/projects/infomeasure/badge/)](https://infomeasure.readthedocs.io/)</a>
<a href="">[![PyPI Version](https://badge.fury.io/py/infomeasure.svg)](https://pypi.org/project/infomeasure/)</a>
<a href="">[![Python Version](https://img.shields.io/pypi/pyversions/infomeasure)](https://pypi.org/project/infomeasure/)</a>
<a href="">[![Anaconda Version](https://anaconda.org/conda-forge/infomeasure/badges/version.svg)](https://anaconda.org/conda-forge/infomeasure)</a>
<a href="">[![PyPI Downloads](https://static.pepy.tech/badge/infomeasure)](https://pepy.tech/projects/infomeasure)</a>

</div>

<div align="center">

<a href="">[![Sci. Rep.](https://img.shields.io/static/v1?label=Sci.%20Rep.&message=10.1038/s41598-025-14053-5&color=005b96)](https://doi.org/10.1038/s41598-025-14053-5)</a>
<a href="">[![Cited by](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.openalex.org%2Fworks%2FW4413108477%3Fselect%3Dcited_by_count&query=%24.cited_by_count&label=cited%20by&suffix=%2B&color=005b96)](https://www.semanticscholar.org/paper/eab2671b69dae18ed1075195d700bbb98dc4712d)</a>
<a href="">[![arXiv](https://img.shields.io/badge/arXiv-2505.14696-b31b1b.svg)](https://arxiv.org/abs/2505.14696)</a>
<a href="">[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15241810.svg)](https://doi.org/10.5281/zenodo.15241810)</a>
<a href="">[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.2-4baaaa.svg)](CODE_OF_CONDUCT.md)</a>

</div>

<div align="center">

<a href="">[![pipeline status](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/badges/main/pipeline.svg)](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/-/commits/main)</a>
<a href="">[![coverage report](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/badges/main/coverage.svg)](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/-/jobs)</a>

</div>

> [!IMPORTANT]
> ⚡ **Rust implementation now available!**
> All core measures have been reimplemented in
> [`infomeasure-rs`](https://crates.io/crates/infomeasure)
> ([crate docs](https://docs.rs/infomeasure)) with
> compile-time type safety, GPU acceleration, and even faster execution.
> Check out the [Rust Guide](https://docs.rs/infomeasure/latest/infomeasure/guide/index.html)
> if you need maximum performance for production or large-scale analysis.
> Find the [benchmark and interactive Rust vs Python performance comparison here](https://cbueth.codeberg.page/infomeasure-rs/).

Continuous and discrete entropy and information measures using different estimation
techniques.

---

For details on how to use this package, see the
[Guide](https://infomeasure.readthedocs.io/en/latest/guide/) or
the [Documentation](https://infomeasure.readthedocs.io/).

## Cited in research

`infomeasure` is used across research fields (selection, as of August 2026):

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

## Setup

This package can be installed from PyPI using pip:

```bash
pip install infomeasure
```

This will automatically install all the necessary dependencies as specified in the
`pyproject.toml` file. It is recommended to use a virtual environment, e.g. using
`conda`, `mamba` or `micromamba` (they can be used interchangeably).
`infomeasure` can be installed from the `conda-forge` channel.

```bash
conda create -n im_env -c conda-forge python
conda activate im_env
conda install -c conda-forge infomeasure
```

## Development Setup

For development, we recommend using `micromamba` to create a virtual
environment (`conda` or `mamba` also work)
and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the desired python version and the dependencies.

```bash
micromamba create -n im_env -c conda-forge python
micromamba activate im_env
```

To let `micromamba` handle the dependencies, use the `requirements` files

```bash
micromamba install -f requirements/build_requirements.txt \
  -f requirements/linter_requirements.txt \
  -f requirements/test_requirements.txt \
  -f requirements/doc_requirements.txt
pip install --no-build-isolation --no-deps -e .
```

Alternatively, if you prefer to use `pip`, installing the package in editable mode will
also install the
development dependencies.

```bash
pip install -e ".[all]"
```

Now, the package can be imported and used in the python environment, from anywhere on
the system if the environment is activated.
For new changes, the repository only needs to be updated, but the package does not need
to be reinstalled.

## Set up Jupyter kernel

If you want to use `infomeasure` with its environment `im_env` in Jupyter, run:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=im_env
```

This allows you to run Jupyter with the kernel `im_env` (Kernel > Change Kernel >
im_env)

## Acknowledgments

This project has received funding from the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation programme (grant agreement No
851255).
This work was partially supported by the María de Maeztu project CEX2021-001164-M funded
by the MICIU/AEI/10.13039/501100011033 and FEDER, EU.
