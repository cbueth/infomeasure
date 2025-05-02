---
file_format: mystnb
kernelspec:
  name: python3
---
(getting_started)=
# Getting Started

This package can be [installed from PyPI](https://pypi.org/project/infomeasure/) using pip:

```bash
pip install infomeasure
```

This will automatically install all the necessary dependencies as specified in the
[`pyproject.toml`](https://github.com/cbueth/infomeasure/blob/main/pyproject.toml) file.
It is recommended to use a virtual environment, e.g., using
[`conda`](https://conda.io/projects/conda/en/latest),
[`mamba`](https://mamba.readthedocs.io/en/latest) or
[`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
(they can be used interchangeably).
`infomeasure` can be installed from
the [`conda-forge`](https://anaconda.org/conda-forge/infomeasure) channel.

```bash
conda create -n im_env -c conda-forge python=3.13
conda activate im_env
conda install -c conda-forge infomeasure
```

## Usage

For a more complete introduction, find the {ref}`reference guide` with
the {ref}`theoretical introduction <introduction>`, and
for a quick start on how to use the estimators,
find the {ref}`Estimator Usage` section.
The most common functions are exposed in the {ref}`top-level namespace <functions>`,
e.g. {py:func}`~infomeasure.entropy` or {py:func}`~infomeasure.estimator`.
For example:

```{code-cell}
import infomeasure as im

data = [0, 1, 0, 1, 0, 1, 0, 1]
entropy = im.entropy(data, approach="kernel", bandwidth=3, kernel="box")
# or
est = im.estimator(data, measure="entropy", approach="kernel", bandwidth=3, kernel="box")
print(f"Entropy with im.entropy   = {entropy}")
print(f"Entropy with im.estimator = {est.result()}")
```

For mutual information, there is a similar function:

```{code-cell}
data_x = [0, 1, 0, 1, 0, 1, 0, 1]
data_y = [0, 1, 0, 1, 0, 0, 0, 0]
mi = im.mutual_information(data_x, data_y, approach="discrete")
# or
est = im.estimator(data_x, data_y, measure="mutual_information",
                   approach="discrete", prop_time=1)
print(f"Mutual Information with im.mutual_information = {mi}")
print(f"Mutual Information with im.estimator          = {est.result()}")
print(f"P-value = {est.p_value(10)}, t-score = {est.t_score(10)}")
```

Transfer entropy can be calculated as follows:

```{code-cell}
source = [0.0, 0.3, 0.5, 1.2, 0.0, 0.4, 0.2, -0.6, -0.8, -0.4]
dest = [0.0, 0.8, -0.7, 0.2, 1.2, 1.0, 1.3, 0.7, 0.8, -0.1]
te = im.transfer_entropy(source, dest, approach="metric", noise_level=0.001)
# or
est = im.estimator(source, dest, measure="transfer_entropy", approach="metric", noise_level=0.001)
#te, (est.result(), est.p_value(10), est.t_score(10))
print(f"Transfer Entropy with im.transfer_entropy = {te}")
print(f"Transfer Entropy with im.estimator        = {est.result()} (differs due to noise)")
print(f"p-value = {est.p_value(10)}, t-score = {est.t_score(10)}")
```

In {ref}`Estimator Usage`, you can find more information on how to use the estimators, specific functions, p-value estimation and which approaches are available.

For more insight into the package, read the {ref}`Reference Guide`
or the {ref}`API Reference`.


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
the system, if the environment is activated.
For new changes, the repository only needs to be updated, but the package does not need
to be reinstalled.
