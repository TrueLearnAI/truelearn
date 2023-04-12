# noqa
"""
DotPlotter Example
==================

This example shows how to use the ``DotPlotter`` class
to generate a bar chart to study the estimated mean of
the learner's knowledge and our confidence level (via error bars).

It is similar to the previous example of the ``BarPlotter``.
The difference is that the bars have been replaced with dots.

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

plotter = visualisations.DotPlotter()

# you can control whether to include history data
# in the plot. If you use `history=True`, when you hover
# your mouse over the dot, you can see statistics about
# the total number of videos watched and the time the learner watched the last video
plotter.plot(classifier.get_learner_model().knowledge, top_n=10, history=True)

# you can also use plotter.show()
# which is a shorthand for calling pio
pio.show(plotter.figure)
