# noqa
"""
RadarPlotter Example
====================

This example shows how to use the ``RadarPlotter`` class
to generate a radar plot that represents the learner's knowledge.

In this example, we use the ``KnowledgeClassifier`` to build up
the knowledge representation of the learner. You could also use
other classifiers like ``NoveltyClassifier``.
"""
from truelearn import learning, datasets
from truelearn.utils import visualisations


data, _, _ = datasets.load_peek_dataset(test_limit=0)

# select a learner from data
_, learning_events = data[12]

classifier = learning.KnowledgeClassifier()
for event, label in learning_events:
    classifier.fit(event, label)

visualisations.RadarPlotter().plot(
    classifier.get_learner_model().knowledge, top_n=10
).show()
