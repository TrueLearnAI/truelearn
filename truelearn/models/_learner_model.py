from __future__ import annotations

from ._abstract_knowledge import AbstractKnowledge
from ._knowledge import Knowledge


class LearnerModel:
    """A Classifier that always makes positive prediction.

    Parameters
    ----------
    knowledge: Knowledge | None
        The representation of the learner's knowledge.
        A new Knowledge will be created if the knowledge is None.
    tau: float
        A dynamic factor of the learner's learning process.

    Attributes
    ----------
    knowledge
    tau

    """

    def __init__(self, knowledge: AbstractKnowledge | None = None, tau: float = 0.1) -> None:
        if knowledge is None:
            self.__knowledge = Knowledge()
        else:
            self.__knowledge = knowledge
        self.__tau = tau

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
