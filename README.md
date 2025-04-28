<div style="text-align: center; max-width: 700px; margin: 0 auto;">
  <img src="https://github.com/cbueth/infomeasure/blob/main/docs/_static/im_logo_transparent.png?raw=true" style="max-width: 100%; height: auto;" alt="infomeasure logo">
</div>

<div style="text-align: center;">

  [![Documentation](https://readthedocs.org/projects/infomeasure/badge/)](https://infomeasure.readthedocs.io/)
  [![PyPI Version](https://badge.fury.io/py/infomeasure.svg)](https://pypi.org/project/infomeasure/)
  [![Python Version](https://img.shields.io/pypi/pyversions/infomeasure)](https://pypi.org/project/infomeasure/)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.2-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>

<div style="text-align: center;">

  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15241810.svg)](https://doi.org/10.5281/zenodo.15241810)
  [![Anaconda Version](https://anaconda.org/conda-forge/infomeasure/badges/version.svg)](https://anaconda.org/conda-forge/infomeasure)

</div>

<div style="text-align: center;">

  [![pipeline status](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/badges/main/pipeline.svg)](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/-/commits/main)
  [![coverage report](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/badges/main/coverage.svg)](https://gitlab.ifisc.uib-csic.es/carlson/infomeasure/-/jobs)

</div>

Continuous and discrete entropy and information measures using different estimation techniques.

---

For details on how to use this package, see the
[Guide](https://infomeasure.readthedocs.io/en/latest/guide/) or
the [Documentation](https://infomeasure.readthedocs.io/).

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
conda create -n im_env -c conda-forge python=3.13
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
micromamba create -n im_env -c conda-forge python=3.13
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

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 851255).
This work was partially supported by the María de Maeztu project CEX2021-001164-M funded by the MICIU/AEI/10.13039/501100011033 and FEDER, EU.
