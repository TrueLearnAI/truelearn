# noqa
"""
LinePlotter Example
===================

This example shows how to use the ``LinePlotter`` class
to generate a bar chart to compare different learners' knowledge
on the same topic.

In this example, we use the ``NoveltyClassifier`` to build
a representation of the learners' knowledge. You could also use
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

# select 2 learners from data
learning_events_for_three_learners = [
    learning_events for _, learning_events in [data[12], data[14]]
]

classifiers = [
    learning.NoveltyClassifier() for _ in range(len(learning_events_for_three_learners))
]
for classifier, learning_events in zip(classifiers, learning_events_for_three_learners):
    for event, label in learning_events:
        classifier.fit(event, label)

plotter = visualisations.LinePlotter()

# you can optionally set title via `title` method
plotter.title("Comparison of topic across different users")

# you can use line plotter to compare the knowledge across
# different learners.
#
# You need to pass these knowledges in a list.
# and specify the topics you want to compare.
# We usually only put one topic inside ``topics`` list
# because we want to compare cross-sectionally how
# different learners' knowledge of the same topic changes over time
plotter.plot(
    [classifier.get_learner_model().knowledge for classifier in classifiers],
    topics=["Expected value"],
)

# you can also use plotter.show()
# which is a shorthand for calling pio
pio.show(plotter.figure)
