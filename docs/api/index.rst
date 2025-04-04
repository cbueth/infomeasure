API Reference
=============

These pages contain the API documentation for `infomeasure`.
For an overview of the package, see the :ref:`Overview <overview>` page.

The idea of this package is based on estimating information-theoretic measures,
such as Entropy, Mutual Information, and Transfer Entropy.
For these we use :py:mod:`Abstract Base Classes <abc>` to define the interface of the estimators.
Some mixins are also provided to provide additional functionality to certain estimators.

.. inheritance-diagram:: infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator infomeasure.estimators.entropy.kernel.KernelEntropyEstimator infomeasure.estimators.entropy.kozachenko_leonenko.KozachenkoLeonenkoEntropyEstimator infomeasure.estimators.entropy.ordinal.OrdinalEntropyEstimator infomeasure.estimators.entropy.renyi.RenyiEntropyEstimator infomeasure.estimators.entropy.tsallis.TsallisEntropyEstimator infomeasure.estimators.mutual_information.discrete.DiscreteMIEstimator infomeasure.estimators.mutual_information.discrete.DiscreteCMIEstimator infomeasure.estimators.mutual_information.kernel.KernelMIEstimator infomeasure.estimators.mutual_information.kernel.KernelCMIEstimator infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGMIEstimator infomeasure.estimators.mutual_information.kraskov_stoegbauer_grassberger.KSGCMIEstimator infomeasure.estimators.mutual_information.ordinal.OrdinalMIEstimator infomeasure.estimators.mutual_information.ordinal.OrdinalCMIEstimator infomeasure.estimators.mutual_information.renyi.RenyiMIEstimator infomeasure.estimators.mutual_information.renyi.RenyiCMIEstimator infomeasure.estimators.mutual_information.tsallis.TsallisMIEstimator infomeasure.estimators.mutual_information.tsallis.TsallisCMIEstimator infomeasure.estimators.transfer_entropy.discrete.DiscreteTEEstimator infomeasure.estimators.transfer_entropy.discrete.DiscreteCTEEstimator infomeasure.estimators.transfer_entropy.kernel.KernelTEEstimator infomeasure.estimators.transfer_entropy.kernel.KernelCTEEstimator infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGTEEstimator infomeasure.estimators.transfer_entropy.kraskov_stoegbauer_grassberger.KSGCTEEstimator infomeasure.estimators.transfer_entropy.ordinal.OrdinalTEEstimator infomeasure.estimators.transfer_entropy.ordinal.OrdinalCTEEstimator infomeasure.estimators.transfer_entropy.renyi.RenyiTEEstimator infomeasure.estimators.transfer_entropy.renyi.RenyiCTEEstimator infomeasure.estimators.transfer_entropy.tsallis.TsallisTEEstimator infomeasure.estimators.transfer_entropy.tsallis.TsallisCTEEstimator
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
