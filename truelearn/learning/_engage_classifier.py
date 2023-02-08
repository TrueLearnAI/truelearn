from __future__ import annotations

from ._base import BaseClassifier
from truelearn.models import EventModel


class EngageClassifier(BaseClassifier):
    """A Classifier that always makes positive prediction.

    Methods
    -------
    fit(x, y)
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage in the given learning event.
    predict_proba(x)
        Predict the probability of learner engagement in the given learning event.

    """

    def fit(self, _x: EventModel, _y: bool) -> EngageClassifier:
        """Train the model based on the given event and labels.

        Parameters
        ----------
        _x : EventModel
            A representation of a learning event.
        _y : bool
            A label that is either True or False.

        Returns
        -------
        EngageClassifier
            The updated model.

        Notes
        -----
        Given the nature of this classifier, the input _x and _y are not used.

        """
        return self

    def predict(self, _x: EventModel) -> bool:
        """Predict whether the learner will engage in the given learning event.

        Parameters
        ----------
        _x : EventModel
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

    def predict_proba(self, _x: EventModel) -> float:
        """Predict the probability of learner engagement in the given learning event.

        Parameters
        ----------
        _x : EventModel
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
