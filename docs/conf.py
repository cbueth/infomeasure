# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# autodoc_mock_imports = [
#     "modules that are not installed but referenced in the docs",
# ]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

project = "infomeasure"
copyright = f"2024–{datetime.now().year}, infomeasure maintainers"
author = "Carlson Büth, Acharya Kishor, and Massimiliano Zanin"
version = "0.5.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "myst_nb",  # Parse and execute ipynb files in Sphinx
    "numpydoc",  # Automatically loads .ext.autosummary
    # "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx_automodapi.automodapi",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
    "sphinx_design",  # Further directives
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

bibtex_bibfiles = ["refs.bib"]

inheritance_graph_attrs = dict(size='""', fontsize=14, ratio="compress", dpi=300)

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
master_doc = "index"

language = "en"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3
myst_links_external_new_tab = True
nb_execution_timeout = 180
nb_execution_excludepatterns = ["Schreiber_Article.ipynb", "Time_Performance.ipynb"]

numpydoc_xref_param_type = True
numpydoc_show_inherited_class_members = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_js_files = ["custom.js"]
html_theme_options = {
    "repository_url": "https://github.com/cbueth/infomeasure",
    "repository_branch": "main",
    "use_source_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
    # "home_page_in_toc": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",  # not supported with gitlab
        "deepnote_url": "https://deepnote.com/",  # not supported with gitlab
        "notebook_interface": "jupyterlab",
    },
    "logo": {
        "image_light": "_static/im_logo_transparent.png",
        "image_dark": "_static/im_logo_transparent_dark.png",
    },
    #  "icon_links": [
    #      {
    #          "name": "GitHub",
    #          "url": "https://github.com/cbueth/infomeasure",  # required
    #          # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
    #          "icon": "fa-brands fa-github",
    #          "type": "fontawesome",
    #      },
    #      {
    #          "name": "PyPI",
    #          "url": "https://pypi.org/project/infomeasure/",
    #          "icon": "https://img.shields.io/pypi/v/infomeasure",
    #          "type": "url",
    #      },
    # ]
}
html_favicon = "_static/im_icon_transparent-200x200.png"
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
        "sidebar-ethical-ads.html",
    ],
    "index": [],
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sparse": ("https://sparse.pydata.org/en/stable", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
