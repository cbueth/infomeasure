API Reference
=============

These pages contain the API documentation for `infomeasure`.
For an overview of the package, see the :ref:`Overview <overview>` page.

The idea of this package is based on estimating information-theoretic measures,
such as Entropy, Mutual Information, and Transfer Entropy.
For these we use :py:mod:`Abstract Base Classes <abc>` to define the interface of the estimators.
Some mixins are also provided to provide additional functionality to certain estimators.

.. inheritance-diagram:: infomeasure.measures.entropy.discrete.DiscreteEntropyEstimator infomeasure.measures.entropy.kernel.KernelEntropyEstimator infomeasure.measures.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator infomeasure.measures.mutual_information.discrete.DiscreteMIEstimator infomeasure.measures.mutual_information.kernel.KernelMIEstimator infomeasure.measures.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator infomeasure.measures.transfer_entropy.discrete.DiscreteTEEstimator infomeasure.measures.transfer_entropy.kernel.KernelTEEstimator infomeasure.measures.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator
   :parts: 1
   :caption: Inheritance diagram for the estimators

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
