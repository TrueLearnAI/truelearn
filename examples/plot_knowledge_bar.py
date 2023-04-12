# noqa
"""
BarPlotter Example
==================

This example shows how to use the ``BarPlotter`` class
to generate a bar chart to study the estimated mean of
the learner's knowledge and our confidence level (via error bars).

In this example, we use the ``NoveltyClassifier`` to build
a representation of the learner's knowledge. You could also use
other classifiers like ``KnowledgeClassifier`` (for building
knowledge representation) or ``InterestClassifier`` (for building
interest representation).
"""
from truelearn import learning, datasets
from truelearn.utils import visualisations

import plotly.io as pio

data, _, _ = datasets.load_peek_dataset(test_limit=0, verbose=False)

# select a learner from data
_, learning_events = data[12]

classifier = learning.NoveltyClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

plotter = visualisations.BarPlotter()

plotter.plot(classifier.get_learner_model().knowledge, top_n=10)

# you can also use plotter.show()
# which is a shorthand for calling pio
pio.show(plotter.figure)
