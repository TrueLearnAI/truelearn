import dataclasses
from typing import NamedTuple

from ._knowledge import Knowledge


@dataclasses.dataclass
class LearnerModel:
    """The model of a learner.

    Examples:
        >>> from truelearn.models import LearnerModel, KnowledgeComponent
        >>> # construct an empty learner model
        >>> LearnerModel()
        LearnerModel(knowledge=Knowledge(knowledge={}), number_of_engagements=0, \
number_of_non_engagements=0)
        >>> # construct a learner model with given engagement stats
        >>> LearnerModel(number_of_engagements=10, number_of_non_engagements=2)
        LearnerModel(knowledge=Knowledge(knowledge={}), number_of_engagements=10, \
number_of_non_engagements=2)
        >>> # construct a learner model with given knowledge
        >>> knowledge = Knowledge({1: KnowledgeComponent(mean=0.0, variance=1.0)})
        >>> LearnerModel(knowledge=knowledge)  # doctest:+ELLIPSIS
        LearnerModel(knowledge=Knowledge(knowledge={1: KnowledgeComponent(mean=0.0, \
variance=1.0, ...)}), number_of_engagements=0, number_of_non_engagements=0)
    """

    knowledge: Knowledge = dataclasses.field(default_factory=Knowledge)
    """A representation of the learner's knowledge."""

    number_of_engagements: int = 0
    """An int indicating how many educational resources learners are engaged with."""

    number_of_non_engagements: int = 0
    """An int indicating how many educational resources learners are not engaged 
    with."""


@dataclasses.dataclass
class LearnerMetaWeights:
    """Store the weights used in meta training.

    Examples:
        >>> from truelearn.models import LearnerMetaWeights
        >>> # construct an empty learner meta model
        >>> LearnerMetaWeights()  # doctest:+ELLIPSIS
        LearnerMetaWeights(novelty_weights=Weights(mean=0.5, variance=0.5)...)
        >>> # construct a learner meta model with custom weights
        >>> bias_weights = LearnerMetaWeights.Weights(mean=1.0, variance=2.0)
        >>> LearnerMetaWeights(bias_weights=bias_weights)
        LearnerMetaWeights(...bias_weights=Weights(mean=1.0, variance=2.0))
    """

    class Weights(NamedTuple):
        """A namedtuple that represents the weights used in LearnerMetaWeights."""

        mean: float
        variance: float

    novelty_weights: Weights = Weights(0.5, 0.5)
    """A dict that stores the "mean" and "variance" of the learner's 
    knowledge/novelty weights."""

    interest_weights: Weights = Weights(0.5, 0.5)
    """"A dict that stores the "mean" and "variance" of the learner's interest 
    weights."""

    bias_weights: Weights = Weights(0.0, 0.5)
    """A dict that stores the "mean" and "variance" of a bias variable."""
