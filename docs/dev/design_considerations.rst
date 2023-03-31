.. _design:

Design Considerations
=====================

Target Platform
---------------
The library is designed to be used across multiple platforms, including Windows, Linux, and Mac OS X.


Docstring format
----------------
We use Google style docstrings for all functions and methods. See the `Google Style Guide`_ for more information.

.. _Google Style Guide: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

We believe that all the methods and classes in TrueLearn should be documented as we believe this will not only help users but also future developers.
So, before you make a PR, make sure you properly document all methods and classes based on the `Google Style Guide`_.


Code Style
----------
We enforce most of PEP8 in our codebase, including but not limited to:

* Imports: Import modules at the top of the file, and use separate lines for each import.
* Naming conventions: Use descriptive names for variables, functions, and modules. Use lowercase letters for variables and functions, and capitalize the first letter of class names.
* Indentation: Use 4 spaces for indentation instead of tabs.
* Method arguments: Use self for the first argument in instance methods, and use cls for the first argument in class methods.

We additionally enforce some other styles:

* Line length: Limit lines to a maximum of 88 characters to ensure readability. (black formatter)
* Imports: Follow the import ordering described in `Google Style Guide`_.


Class Design
------------

Design Principles
^^^^^^^^^^^^^^^^^
Before we started designing TrueLearn, we had a clear goal in mind: TrueLearn should be easy to use and easy to extend.

To decide what it would look like, we reviewed the API design of a well-known Python machine learning library `scikit-learn`_.

.. _scikit-learn: https://github.com/scikit-learn/scikit-learn

Based on their paper `API design for machine learning software: experiences from the scikit-learn project <https://arxiv.org/abs/1309.0238>`_,
this was achieved by adhering to various design principles:

* Consistency: “All objects share a consistent interface composed of a limited set of methods.”
* Inspection: “Constructor parameters and parameter values determined by learning algorithms are stored and exposed as public attributes.”
* Non-proliferation of classes: “Learning algorithms are the only objects to be represented using custom classes. Datasets are represented as NumPy arrays or SciPy sparse matrices. Hyper-parameter names and values are represented as standard Python strings or numbers whenever possible.”
* Composition: “Whenever feasible, meta-algorithms parametrized on other algorithms are implemented and composed from existing building blocks.”
* Sensible defaults: “Whenever an operation requires a user-deﬁned parameter, an appropriate default value is deﬁned by the library.”

These five design principles are reflected in the three main interfaces of scikit-learn:
``estimator``, ``predictor`` and ``transformer``.
The estimator interface provides consistent interfaces for training the model (`fit`) and
public attributes (`coef_`) for inspection of the internal states of the model.
The predictor interface provides consistent interfaces for utilizing the model (`predict`, `predict_prob`)
and assessing the performance of the model (`score`).
The existence of transformer interface (`transform` method) makes it easy for users to perform common preprocessing on their data.
**We decide to mimic this structure for our classifier**.

In terms of data representation of scikit-learn,
“datasets are encoded in NumPy multidimensional arrays for dense data and as SciPy sparse matrices for sparse data”.
This allows scikit-learn to utilize the efficient Numpy and SciPy operations while writing readable and maintainable code.
For TrueLearn, the only problem with the above representation is that “the public interface is oriented towards processing batches of samples”,
but our library expects data to be more “discrete” because the user engagement with some videos is not likely obtained in large batches.
**Therefore, we will provide functions that use on a single piece of data**.

Finally, it is worth mentioning what makes scikit-learn extensible and its code more reusable.
Apart from the interface design mentioned above, scikit-learn's extensibility comes mainly from duck typing,
which is based on the idea that "If it walks like a duck and it quacks like a duck, then it must be a duck"
This means that one can use their model with existing implementations of the library,
if they design their model according to the three implicit interfaces mentioned above.
This provides a lot of flexibility for developers as they “are not forced to inherit from any scikit-learn class”.
**The idea of duck typing is used in many places in TrueLearn (in a more friendly format)**.

But they have now made some changes to the design. Their ``Classifier`` implementations now inherit a class ``BaseClassifier`` to provide
parameter type checking and implementation of some common methods such as ``__repr__``.
**TrueLearn took inspiration from it and designed our ``BaseClassifier``, type checking and implemented common methods.**


Project Structure
^^^^^^^^^^^^^^^^^
TrueLearn is structured as a package containing several subpackages.

At time of writing, the repository includes the following subpackages:

* truelearn/datasets: contains the methods to download and load PEEKDataset.
* truelearn/learning: contains the implementation of all the classifiers.
* truelearn/models: contains the definitions of event model and learner model.
* truelearn/preprocessing: contains the preprocessing utilities, such as wikifier.
* truelearn/tests: contains all the tests for TrueLearn library.
* truelearn/utils: contains two utility packages, ``metrics`` (contains scoring functions) and ``visualization`` (contain different visualizations).


truelearn.models
^^^^^^^^^^^^^^^^
The ``truelearn.models`` package is made up from several import parts

* base: contains the basic building blocks of learner and event knowledge.
  We use an ontology based on Wikipedia to represent KCs, where each Wikipedia page is considered as an independent and atomic unit of knowledge (i.e. a KC).
  ``BaseKnowledgeComponent`` is the base classifier for all knowledge components. You can inherit this to design new ``KnowledgeComponent``.
* knowledge: We have already defined two knowledge components ``KnowledgeComponent`` and ``HistoryAwareKnowledgeComponent`` based on ``BaseKnowledgeComponent``.
  It also contains a class ``Knowledge`` that represents the learner and event knowledge.
* learner: we define ``LearnerModel`` to represent the learner and ``LearnerMetaWeights`` to represent the weights of different learner models when the developer
  uses meta-learning (``INKClassifier``).
* event: we define ``EventModel`` to represent the event.


truelearn.learning
^^^^^^^^^^^^^^^^^^
The ``truelearn.learning`` package consists of different classifiers in `TrueLearn: A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources`_.

.. _TrueLearn\: A Family of Bayesian Algorithms to Match Lifelong Learners to Open Educational Resources: https://arxiv.org/abs/1911.09471

* Base Classifiers: baseline classifiers in paper.
  It contains ``EngageClassifier``, ``PersistentClassifier``, ``MajorityClassifier``.
* ``KnowledgeClassifier``: utilize the fixed-depth representation of event knowledge and rely on the third assumption in the first TrueLearn paper.
* ``NoveltyClassifier``: utilize the fourth assumption in the TrueLearn paper.
* ``InterestClassifier``: learn based on the learner's interest described in `Power to the Learner: Towards Human-Intuitive and Integrative Recommendations with Open Educational Resources`_.
* ``INKClassifier``: use ``NoveltyClassifier`` and ``InterestClassifier`` for meta-learning.

.. _Power to the Learner\: Towards Human-Intuitive and Integrative Recommendations with Open Educational Resources: https://www.mdpi.com/2071-1050/14/18/11682

If your goal is to support new classifiers, you can start here. We welcome PR to add new classifiers,
such as those that make use of knowledge tracking.
If you plan to do this, please discuss it with us as we may need to restructure this sub-package.


truelearn.utils.visualisations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO


truelearn.tests
^^^^^^^^^^^^^^^
This package contains all the tests for TrueLearn.

* test_datasets: contains the tests for ``truelearn.datasets``.
* test_learning: contains the tests for ``truelearn.learning``.
* test_models: contains the tests for ``truelearn.models``.
* test_datasets: contains the tests for ``truelearn.datasets``.
* test_datasets: contains the tests for ``truelearn.datasets``.
* test_datasets: contains the tests for ``truelearn.datasets``.

To learn how to run the tests and add more tests to TrueLearn, please refer to :ref:`testing`.
