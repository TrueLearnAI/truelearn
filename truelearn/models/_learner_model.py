from __future__ import annotations

from ._knowledge import Knowledge


# TODO: update documentation tomorrow
class LearnerModel:
    """A Classifier that always makes positive prediction.

    Parameters
    ----------
    knowledge: Knowledge
    tau: float
    beta: float

    Attributes
    ----------
    knowledge
    tau
    beta

    """

    def __init__(self, knowledge: Knowledge | None = None, tau: float = 0., beta: float = 0.) -> None:
        self.__knowledge = knowledge
        if self.__knowledge is None:
            self.__knowledge = Knowledge()
        self.__tau = tau
        self.__beta = beta

    @property
    def knowledge(self):
        return self.knowledge

    @property
    def tau(self):
        return self.__tau

    @property
    def beta(self):
        return self.__beta
