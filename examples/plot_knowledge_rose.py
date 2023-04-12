# noqa
"""
RosePlotter Example
===================

This example shows how to use the ``RosePlotter`` class
to generate a rose pie chart to study the distribution of
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
# if it follows the protocol of history aware knowledge component
data, _, _ = datasets.load_peek_dataset(
    test_limit=0, kc_init_func=models.HistoryAwareKnowledgeComponent, verbose=False
)

# select a learner from data
_, learning_events = data[12]

classifier = learning.NoveltyClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

plotter = visualisations.RosePlotter()

# you can control whether to include all other data
# in a section called other.
plotter.plot(classifier.get_learner_model().knowledge, top_n=5, other=True)

# you can also use plotter.show()
# which is a shorthand for calling pio
pio.show(plotter.figure)
