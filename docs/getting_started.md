---
file_format: mystnb
kernelspec:
  name: python3
---

# Getting Started

> Until public, use the [Development Setup](#development-setup)!

This package can be installed from PyPI using pip:

```bash
pip install infomeasure
```

This will automatically install all the necessary dependencies as specified in the
[`pyproject.toml`](https://github.com/cbueth/infomeasure/blob/main/pyproject.toml) file.
It is recommended to use a virtual environment, e.g. using
[`conda`](https://conda.io/projects/conda/en/latest),
[`mamba`](https://mamba.readthedocs.io/en/latest) or
[`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
(they can be used interchangeably).
`infomeasure` can be installed from the `conda-forge` channel _(not yet)_.

```bash
conda create -n im_env -c conda-forge python=3.12
conda activate im_env
conda install -c conda-forge infomeasure  # when feedstock is available
```

## Usage

The package can be used as a library.
The most common functions are exposed in the top-level namespace,
e.g. {py:func}`~infomeasure.entropy` and
{py:mod}`~infomeasure.estimators.kde`. The latter can also be imported directly from the
submodule
[...] For example:

```{code-cell}
import infomeasure as im

data = [0, 1, 0, 1, 0, 1, 0, 1]
entropy = im.entropy(data, estimator="kde")
# or
entropy = im.estimators.kde.entropy(data)
```

[...]

[//]: # (TODO: Also show MI and TE examples)

For more insight into the package, read the [Guide](guide/index.myst)
or the [API Reference](api/index.rst).


### Set up Jupyter kernel

If you want to use `infomeasure` with its environment `im_env` in Jupyter, run:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=im_env
```

This allows you to run Jupyter with the kernel `im_env` (Kernel > Change Kernel >
im_env)


## Development Setup

For development, we recommend using [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
to create a virtual environment (`conda` or `mamba` also work)
and installing the package in editable mode.
After cloning the repository, navigate to the root folder and
create the environment with the desired python version and the dependencies.

```bash
micromamba create -n im_env -c conda-forge python=3.12
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
