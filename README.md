# `infomeasure` — Information Measure Estimators

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://carlson.pages.ifisc.uib-csic.es/infomeasure/)
[![pipeline status](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/badges/main/pipeline.svg)](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/-/pipelines?page=1&scope=all&ref=main)
[![coverage report](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/badges/main/coverage.svg)](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/-/commits/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.2-4baaaa.svg)](CODE_OF_CONDUCT.md)

Continuous and discrete entropy and information measures using different estimators.

---

For details on how to use this package, see the
[Guide](https://carlson.pages.ifisc.uib-csic.es/infomeasure/guide/) or
the [Documentation](https://carlson.pages.ifisc.uib-csic.es/infomeasure/).

## Setup (don't use until public)

This package can be installed from PyPI using pip:

```bash
pip install infomeasure  # when public on PyPI
```

This will automatically install all the necessary dependencies as specified in the
`pyproject.toml` file. It is recommended to use a virtual environment, e.g. using
`conda`, `mamba` or `micromamba` (they can be used interchangeably).
`infomeasure` can be installed from the `conda-forge` channel (not yet).

```bash
conda create -n im_env -c conda-forge python=3.13
conda activate im_env
conda install -c conda-forge infomeasure  # when feedstock is available
```

## Development Setup

For development, we recommend using `micromamba` to create a virtual
environment (`conda` or `mamba` also work)
and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the desired python version and the dependencies.

```bash
micromamba create -n im_env -c conda-forge python=3.13
micromamba activate im_env
```

To let `micromamba` handle the dependencies, use the `requirements.txt` file

```bash
micromamba install --file requirements.txt
pip install --no-build-isolation --no-deps -e .
```

Alternatively, if you prefer to use `pip`, installing the package in editable mode will
also install the
development dependencies.

```bash
pip install -e ".[all]"
```

Now, the package can be imported and used in the python environment, from anywhere on
the system, if the environment is activated.
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
