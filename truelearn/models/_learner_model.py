from dataclasses import dataclass, field

from ._knowledge import Knowledge


@dataclass
class LearnerModel:
    """A class that models the learner in TrueLearn algorithm.

    Parameters
    ----------
    knowledge: Knowledge | None
        The representation of the learner's knowledge.
        A new Knowledge will be created if the knowledge is None.
    number_of_engagements: int
        The number of engagements with the learnable units.
    number_of_non_engagements: int
        The number of non-engagements with the learnable units.

    Properties
    ----------
    knowledge
    number_of_engagements
    number_of_non_engagements

    # TODO: rename Properties to Attributes after switching to google style

    """

    knowledge: Knowledge = field(default_factory=Knowledge)
    number_of_engagements: int = 0
    number_of_non_engagements: int = 0


@dataclass
class MetaLearnerModel:
    """Placeholder."""

    learner_novelty: LearnerModel = field(default_factory=LearnerModel)
    learner_interest: LearnerModel = field(default_factory=LearnerModel)
    novelty_weight: dict[str, float] = field(default_factory=lambda: {"mean": 0., "variance": 1})
    interest_weight: dict[str, float] = field(default_factory=lambda: {"mean": 0., "variance": 1})
    bias_weight: dict[str, float] = field(default_factory=lambda: {"mean": 0., "variance": 1})
