from __future__ import annotations

from ._abstract_knowledge import AbstractKnowledge
from ._knowledge import Knowledge


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

    def __init__(self, knowledge: AbstractKnowledge | None = None, tau: float = 0.1,
                 number_of_engagements: int = 0, number_of_non_engagements: int = 0) -> None:
        if knowledge is None:
            self.__knowledge = Knowledge()
        else:
            self.__knowledge = knowledge
        self.__tau = tau
        self.__number_of_engagements = number_of_engagements
        self.__number_of_non_engagements = number_of_non_engagements

    @property
    def knowledge(self) -> AbstractKnowledge:
        """Return the Knowledge representation of the learner.

        Returns
        -------
        Knowledge

        """
        return self.__knowledge

    @property
    def tau(self) -> float:
        """Return the dynamic factor of the learner.

        Returns
        -------
        float

        """
        return self.__tau

    @property
    def number_of_engagements(self) -> int:
        """Return the number of the engagements of the learner.

        Returns
        -------
        float

        """
        return self.__number_of_engagements

    @property
    def number_of_non_engagements(self) -> int:
        """Return the number of the non-engagements of the learner.

        Returns
        -------
        float

        """
        return self.__number_of_non_engagements
