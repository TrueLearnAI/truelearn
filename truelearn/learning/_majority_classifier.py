from typing_extensions import Self
from typing import Any

from ._base import BaseClassifier
from truelearn.models import EventModel


class MajorityClassifier(BaseClassifier):
    """A Classifier that predicts based on the number of learner's engagement and non-engagement.

    If the number of engagement on the training data is greater than the number of non-engagement,
    the classifier predicts Engage (True); otherwise, it predicts Non-Engage (False).

    Methods
    -------
    fit(x, y)
        Train the model based on the given event and label.
    predict(x)
        Predict whether the learner will engage in the given learning event.
    predict_proba(x)
        Predict the probability of learner engagement in the given learning event.
    get_params()
        Get the parameters associated with the model.
    set_params(**kargs)
        Set the parameters associated with the model.

    # TODO: remove method section after switching to google style

    """

    _parameter_constraints: dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "_engagement": int,
        "_non_engagement": int
    }

    def __init__(self, engagement: int = 0, non_engagement: int = 0) -> None:
        super().__init__()

        self._engagement = engagement
        self._non_engagement = non_engagement

        self._validate_params()

    def fit(self, _x: EventModel, y: bool) -> Self:
        """Train the model based on the given event and labels.

        Parameters
        ----------
        _x : EventModel
            A representation of a learning event.
        y : bool
            A label that is either True or False.

        Returns
        -------
        Self
            The updated Classifier.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        if y:
            self._engagement += 1
        else:
            self._non_engagement += 1

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
        return self._engagement > self._non_engagement

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
        return self._engagement > self._non_engagement
