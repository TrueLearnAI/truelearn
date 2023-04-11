# noqa
"""
PiePlotter Example
==================

This example shows how to use the ``PiePlotter`` class
to generate a pie chart to study the distribution of
the learner's knowledge.

In this example, we use the ``NoveltyClassifier`` to build
a representation of the learner's knowledge. You could also use
other classifiers like ``KnowledgeClassifier`` (for building
knowledge representation) or ``InterestClassifier`` (for building
interest representation).
"""
from truelearn import learning, datasets, models
from truelearn.utils import visualisations

import plotly.io as pio

# You can also use a custom knowledge component
# if it follows the protocol of base knowledge component
data, _, _ = datasets.load_peek_dataset(
    test_limit=0, kc_init_func=models.HistoryAwareKnowledgeComponent
)

# select a learner from data
_, learning_events = data[12]

classifier = learning.NoveltyClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

plotter = visualisations.PiePlotter()

# you can control whether to include history data
# in the plot. If you use `history=True`, it requires
# the knowledge contains a history attribute.
# This is why we use models.HistoryAwareKnowledgeComponent above
plotter.plot(classifier.get_learner_model().knowledge, top_n=10, history=True)

# you can also use plotter.show()
# which is a shorthand for calling pio
pio.show(plotter.figure)
