from __future__ import annotations
from dataclasses import dataclass, field

from ._abstract_knowledge import AbstractKnowledge
from ._knowledge import Knowledge


@dataclass
class LearnerModel:
    """A class that models the learner in TrueLearn algorithm.

    Parameters
    ----------
    knowledge: Knowledge | None
        The representation of the learner's knowledge.
        A new Knowledge will be created if the knowledge is None.
    tau: float
        A dynamic factor of the learner's learning process.
    number_of_engagements: int
        The number of engagements with the learnable units.
    number_of_non_engagements: int
        The number of non-engagements with the learnable units.

    Properties
    ----------
    knowledge
    tau
    number_of_engagements
    number_of_non_engagements

    """

    knowledge: AbstractKnowledge = field(default_factory=Knowledge)
    tau: float = 0.1
    number_of_engagements: int = 0
    number_of_non_engagements: int = 0
