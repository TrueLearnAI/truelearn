from __future__ import annotations

from ._abstract_knowledge import AbstractKnowledge
from ._knowledge import Knowledge


class EventModel:
    """A class that models a learning event in TrueLearn algorithm.

    Parameters
    ----------
    knowledge: Knowledge | None
        The knowledge representation of the learnable unit.
        A new Knowledge will be created if the knowledge is None.
    event_time: float | None, optional
        The POSIX timestamp when the learning event happens.
        If the timestamp is None, the event timestamp will be None,
        which affects the training of classifiers that utilize Interest.

    Properties
    ----------
    knowledge
    event_time

    """

    def __init__(self, knowledge: AbstractKnowledge | None = None, event_time: float | None = None) -> None:
        if knowledge is None:
            self.__knowledge = Knowledge()
        else:
            self.__knowledge = knowledge
        self.__event_time = event_time

    @property
    def knowledge(self) -> AbstractKnowledge:
        """Return the Knowledge representation of the learner.

        Returns
        -------
        Knowledge

        """
        return self.__knowledge

    @property
    def event_time(self) -> float | None:
        """Return the POSIX timestamp of the event.

        Returns
        -------
        float | None

        """
        return self.__event_time
