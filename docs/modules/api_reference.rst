API Reference
=============

Shown is the class and function hierarchy of TrueLearn.

For a more detailed description of the decisions made during implementation,
please refer to the :ref:`design` section.


:mod:`truelearn.learning`: Contains the classifiers for learning
================================================================
.. automodule:: truelearn.learning
   :no-members:
   :show-inheritance:

Classes
-------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/
    :template: class.rst

    learning.BaseClassifier
    learning.INKClassifier
    learning.InterestClassifier
    learning.KnowledgeClassifier
    learning.NoveltyClassifier
    learning.EngageClassifier
    learning.PersistentClassifier
    learning.MajorityClassifier


:mod:`truelearn.models`: Contains the representations used for learning
=======================================================================
.. automodule:: truelearn.models
   :no-members:
   :show-inheritance:

Classes
-------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/
    :template: class.rst

    models.AbstractKnowledgeComponent
    models.KnowledgeComponent
    models.Knowledge
    models.EventModel
    models.LearnerModel
    models.LearnerMetaModel

:mod:`truelearn.preprocessing`: Contains preprocessing functions and classes
============================================================================
.. automodule:: truelearn.preprocessing
   :no-members:
   :show-inheritance:

Classes
-------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.Annotation
    preprocessing.Wikifier

Functions
---------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/

    preprocessing.get_values_mean
    preprocessing.get_values_sample_std
    preprocessing.get_values_population_std



:mod:`truelearn.utils.metrics`: Contains functions to generate metrics
======================================================================
.. automodule:: truelearn.utils.metrics
   :no-members:
   :show-inheritance:

Functions
---------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/

        utils.metrics.get_precision_score
        utils.metrics.get_recall_score
        utils.metrics.get_accuracy_score
        utils.metrics.get_f1_score

:mod:`truelearn.utils.visualisations`: Contains plotting functions and classes
==============================================================================
.. automodule:: truelearn.utils.visualisations
   :no-members:
   :show-inheritance:

Classes
-------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/
    :template: class.rst

Functions
---------
.. currentmodule:: truelearn
.. autosummary::
    :toctree: generated/