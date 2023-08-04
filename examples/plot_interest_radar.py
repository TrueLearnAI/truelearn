# noqa
"""
RadarPlotter Example
====================

This example shows how to use the ``RadarPlotter`` class
to generate a radar plot to study the mean and variance
of learners' interest in different subjects.

In this example, we use the ``InterestClassifier`` to build
a representation of the learner's interest. You could also use
other classifiers like ``KnowledgeClassifier`` or ``NoveltyClassifier``
to build a representation of learner's knowledge.
"""
from truelearn import learning, datasets
from truelearn.utils import visualisations

import plotly.io as pio

data, _, _ = datasets.load_peek_dataset(test_limit=0, verbose=False)

# select a learner from data
_, learning_events = data[12]

classifier = learning.InterestClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

plotter = visualisations.RadarPlotter()

# you can optionally set a title
plotter.title("Mean and variance of interest in different topics.")

# we could select topics we care via `topics`
plotter.plot(
    classifier.get_learner_model().knowledge,
    topics=[
        "Expected value",
        "Probability",
        "Sampling (statistics)",
        "Calculus of variations",
        "Dimension",
        "Computer virus",
    ],
    visualize_variance=False,
)

# you can also use plotter.show() here
# which is a shorthand for calling pio
pio.show(plotter.figure)
