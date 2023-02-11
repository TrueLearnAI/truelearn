from typing_extensions import Self
from typing import Any

from ._base import BaseClassifier
from truelearn.models import EventModel


class PersistentClassifier(BaseClassifier):
    """A classifier that makes predictions based on whether the learner has engaged with the last learnable unit.

    Methods
    -------
    fit(x, y)
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage in the given learning event.
    predict_proba(x)
        Predict the probability of learner engagement in the given learning event.

    # TODO: remove method section after switching to google style

    """

    _parameter_constraints: dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "_engage_with_last": bool
    }

    def __init__(self, engage_with_last: bool = False) -> None:
        super().__init__()

        self._engage_with_last = engage_with_last

        self._validate_params()

    def fit(self, _x: EventModel, y: bool) -> Self:
        """Train the model based on the given event and labels.

        Parameters
        ----------
        _x : EventModel
            A representation of a learning event.
        y: bool
            A label that is either True or False.

        Returns
        -------
        Self
            The updated Classifier.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        self._engage_with_last = y
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
        return self._engage_with_last

    def predict_proba(self, _x: EventModel) -> float:
        """Predict the probability of learner engagement.

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
        return float(self._engage_with_last)
