from typing import Any
from typing_extensions import Self

from ._base import BaseClassifier
from truelearn.models import EventModel


class KnowledgeClassifier(BaseClassifier):
    """A Knowledge Classifier.

    Parameters
    ----------
    learner_model: LearnerModel | None, optional
    threshold: float
        Threshold for judging learner engagement. If the probability of the learner engagement is greater
        than the threshold, the model will predict engagement.
    init_skill: float
        The initial skill (mean) of the learner given a new AbstractKnowledgeComponent.
    def_var: float
        The default variance of the new AbstractKnowledgeComponent.
    beta: float
        The noise factor, which is used in trueskill.
    positive_only: bool
        Whether the model updates itself only if encountering positive data.

    Methods
    -------
    fit(x, y)
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.
    get_params()
        Get the parameters associated with the model.
    set_params(**kargs)
        Set the parameters associated with the model.

    """

    _parameter_constraints: dict[str, Any] = {
        **BaseClassifier._parameter_constraints
    }

    def __init__(self) -> None:
        self._validate_params()

    def fit(self, x: EventModel, y: bool) -> Self:
        """Train the model based on the given event and labels.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.
        y : bool
            A label that is either True or False.

        Returns
        -------
        Self
            The updated Classifier.

        Notes
        -----
        Given the nature of this classifier, the input _x and _y are not used.

        """
        return self

    def predict(self, _: EventModel) -> bool:
        """Predict whether the learner will engage in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        bool
            Whether the learner will engage in the given learning event.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        return True

    def predict_proba(self, x: EventModel) -> float:
        """Predict the probability of learner engagement in the given learning event.

        Parameters
        ----------
        x : EventModel
            A representation of a learning event.

        Returns
        -------
        float
            The probability that the learner will engage in the given learning event.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        return 1.
