import dataclasses

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
    """

    knowledge: Knowledge = dataclasses.field(default_factory=Knowledge)
    number_of_engagements: int = 0
    number_of_non_engagements: int = 0


@dataclasses.dataclass
class MetaLearnerModel:
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
    """

    learner_novelty: LearnerModel = dataclasses.field(
        default_factory=LearnerModel
    )
    learner_interest: LearnerModel = dataclasses.field(
        default_factory=LearnerModel
    )
    novelty_weight: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "variance": 1}
    )
    interest_weight: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "variance": 1}
    )
    bias_weight: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "variance": 1}
    )
