# noqa
"""
WordPlotter Example
===================

This example shows how to use the ``WordPlotter`` class
to generate a word cloud to study the learner's knowledge.

In this example, we use the ``KnowledgeClassifier`` to build
a representation of the learner's knowledge. You could also use
other classifiers like ``NoveltyClassifier``.

.. note:: You need to use ``WordPlotter`` with caution because

            - Compatibility: it fails to build in Windows for Python version > 3.7.
            - Stability: it is not actively maintained.
"""
from truelearn import learning, datasets
from truelearn.utils import visualisations

# use a custom knowledge component
# you can always use your knowledge component here
# as soon as it follows the protocol of history aware knowledge component
data, _, _ = datasets.load_peek_dataset(test_limit=0, verbose=False)

# select a learner from data
_, learning_events = data[12]

classifier = learning.KnowledgeClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

plotter = visualisations.WordPlotter()

plotter.plot(classifier.get_learner_model().knowledge)

plotter.show()
