import dataclasses
from typing import Dict

from ._knowledge import Knowledge


@dataclasses.dataclass
class LearnerModel:
    """The model of a learner.

    Attributes:
        knowledge:
            A representation of the learner's knowledge.
        number_of_engagements:
            An int indicating how many educational resources learners are engaged with.
        number_of_non_engagements:
            An int indicating how many educational resources learners are not
            engaged with.

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
    number_of_engagements: int = 0
    number_of_non_engagements: int = 0


@dataclasses.dataclass
class LearnerMetaModel:
    """The meta-model of a learner.

    It is modeled by using learner's knowledge/novelty and interests.
    It contains a set of weights that show whether the learners are
    interest-oriented or knowledge/novelty-oriented in the learning process.

    Attributes:
        learner_novelty:
            A learner model that models the learner's knowledge/novelty.
        learner_interest:
            A learner model that models the learner's interest.
        novelty_weight:
            A dict that stores the "mean" and "variance" of the learner's
            knowledge/novelty weights.
        interest_weight:
            A dict that stores the "mean" and "variance" of the learner's
            interest weights.
        bias_weight:
            A dict that stores the "mean" and "variance" of a bias variable.

    Examples:
        >>> from truelearn.models import LearnerModel, LearnerMetaModel
        >>> # construct an empty learner meta model
        >>> LearnerMetaModel()  # doctest:+ELLIPSIS
        LearnerMetaModel(learner_novelty=..., learner_interest=..., \
novelty_weight={'mean': 0.0, 'variance': 1.0}, interest_weight=..., bias_weight=...)
        >>> # construct a learner meta model with given learner models
        >>> learner_novelty = LearnerModel()
        >>> learner_interest = LearnerModel()
        >>> LearnerMetaModel(learner_novelty=learner_novelty,\
learner_interest=learner_novelty)
        LearnerMetaModel(learner_novelty=..., learner_interest=..., \
novelty_weight={'mean': 0.0, 'variance': 1.0}, interest_weight=..., bias_weight=...)
        >>> # construct a learner meta model with custom weights
        >>> LearnerMetaModel(bias_weight={"mean": 1.0, "variance": 2.0})
        LearnerMetaModel(learner_novelty=..., learner_interest=..., \
novelty_weight=..., interest_weight=..., bias_weight={'mean': 1.0, 'variance': 2.0})
    """

    learner_novelty: LearnerModel = dataclasses.field(default_factory=LearnerModel)
    learner_interest: LearnerModel = dataclasses.field(default_factory=LearnerModel)
    novelty_weight: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "variance": 1.0}
    )
    interest_weight: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "variance": 1.0}
    )
    bias_weight: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "variance": 1.0}
    )
