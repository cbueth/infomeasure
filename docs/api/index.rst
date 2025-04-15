API Reference
=============

These pages contain the API documentation for `infomeasure`.
For an overview of the package, see the :ref:`Overview <overview>` page.

The :ref:`infomeasure <infomeasure_docs>` package consists out of one estimator class per approach
and information-theoretic measure.

For convenience, a module level API is also available,
with the following :ref:`top-level functions <functions>`.
Find the :ref:`Estimator Usage <estimator usage>` page for a comprehensive
explanation on how to use them.

Estimators
----------

.. toctree::
   :maxdepth: 1

   entropy/index
   mi/index
   te/index


.. automodapi:: infomeasure
   :no-main-docstr:
   :no-inheritance-diagram:
   :noindex:


Class Inheritance Diagram
-------------------------

The idea of this package is based on estimating information-theoretic measures,
such as Entropy, Mutual Information, and Transfer Entropy.
For these we use :py:mod:`Abstract Base Classes <abc>` to define the interface of the estimators.
Some mixins are also provided to provide additional functionality, e.g., P-values, to certain estimators.

.. inheritance-diagram:: infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator infomeasure.estimators.entropy.kernel.KernelEntropyEstimator infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator infomeasure.estimators.mutual_information.discrete.DiscreteCMIEstimator infomeasure.estimators.mutual_information.kernel.KernelMIEstimator infomeasure.estimators.mutual_information.kernel.KernelCMIEstimator infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGCMIEstimator infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator infomeasure.estimators.mutual_information.ordinal.OrdinalCMIEstimator infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator infomeasure.estimators.mutual_information.renyi.RenyiCMIEstimator infomeasure.estimators.mutual_information.tsallis.TsallisMIEstimator infomeasure.estimators.mutual_information.tsallis.TsallisCMIEstimator infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator infomeasure.estimators.transfer_entropy.discrete.DiscreteCTEEstimator infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator infomeasure.estimators.transfer_entropy.kernel.KernelCTEEstimator infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGCTEEstimator infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator infomeasure.estimators.transfer_entropy.ordinal.OrdinalCTEEstimator infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator infomeasure.estimators.transfer_entropy.renyi.RenyiCTEEstimator infomeasure.estimators.transfer_entropy.tsallis.TsallisTEEstimator infomeasure.estimators.transfer_entropy.tsallis.TsallisCTEEstimator
   :parts: 1
   :caption: Inheritance diagram for all estimators.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
