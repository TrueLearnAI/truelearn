# noqa
"""
TreePlotter Example
===================

This example shows how to use the ``TreePlotter`` class
to generate a treemap to study the distribution of
the learner's knowledge.

In this example, we use the ``KnowledgeClassifier`` to build
a representation of the learner's knowledge. You could also use
other classifiers like ``NoveltyClassifier``.
"""
from truelearn import learning, datasets, models
from truelearn.utils import visualisations

import plotly.io as pio

# use a custom knowledge component
# you can always use your knowledge component here
# as soon as it follows the protocol of history aware knowledge component
data, _, _ = datasets.load_peek_dataset(
    test_limit=0, kc_init_func=models.HistoryAwareKnowledgeComponent, verbose=False
)

# select a learner from data
_, learning_events = data[12]

classifier = learning.KnowledgeClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

plotter = visualisations.TreePlotter()

# you can control whether to include history data
# in the plot. If you use `history=True`, it requires
# the knowledge contains a history attribute.
# This is why we use models.HistoryAwareKnowledgeComponent above
plotter.plot(classifier.get_learner_model().knowledge, top_n=10, history=True)

# you can also use plotter.show()
# which is a shorthand for calling pio
pio.show(plotter.figure)
