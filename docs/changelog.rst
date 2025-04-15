*********
Changelog
*********


Version 0.3.0 (2025-04-01)
**************************

The `0.3.0dev0` release focuses on performance improvements, feature enhancements, and API updates.

- 🔧 **Local values support**: All approaches now support local values.

- 🎯 Added two new composite measures:

  - Jensen-Shannon Divergence (JSD)

  - Kullback-Leibler Divergence (KLD)

- ✨ Optimized algorithms for:

  - Mutual Information (MI) and Conditional Mutual Information (CMI) on discrete and ordinal data.

  - Transfer Entropy (TE) and Conditional Transfer Entropy (CTE).

- ⚡ Major API refactoring to improve compatibility with arbitrary many random variables in MI and CMI.

- 💡 Enhanced performance through optimizations in `base.py`.

- 🔍 Added extensive testing for local values and tested manually with code notebooks.

- ⬆️ Added Python 3.13 support.


Version 0.2.1 (2025-02-11)
**************************

The `0.2.1dev0` release marks the first release, providing essential information
measures and estimators like Entropy (H), Mutual Information (MI), and others.
It includes a CI/CD pipeline, supports Python 3.10-3.12, and is licensed under AGPLv3+.

- 📦 **First release** of the `infomeasure` package.

- 🧩 Added essential information measure estimators:

  - Shannon entropy (H)
  - Mutual Information (MI)
  - Conditional Mutual Information (CMI)
  - Transfer Entropy (TE) and Conditional Transfer Entropy (CTE)
  - Jensen-Shannon Divergence (JSD)
  - Kullback-Leibler Divergence (KLD)

- 🔄 Set up CI/CD pipeline with GitLLab CI.

- 💻 Added support for Python 3.10+.

- 📄 Updated documentation to include installation guide, package structure,
  and example use cases.


Version 0.0.0 (2024-06-06)
**************************

* Package setup

  - 🏗 Written `pyproject.toml`
  - 🔄 General project and test structure with CI/CD
  - 📚️ Documentation with `sphinx`, `sphinxcontrib-bibtex` and `numpydoc`
