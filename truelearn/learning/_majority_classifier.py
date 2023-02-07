from __future__ import annotations

from ._base_classifier import BaseClassifier
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

    Properties
    ----------
    engagement
    non_engagement

    """

    def __init__(self) -> None:
        super().__init__()

        self.__engagement = 0
        self.__non_engagement = 0

    def fit(self, _x: EventModel, y: bool) -> MajorityClassifier:
        """Train the model based on the given event and labels.

        Parameters
        ----------
        _x : EventModel
            A representation of a learning event.
        y : bool
            A label that is either True or False.

        Returns
        -------
        MajorityClassifier
            The updated model.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        if y:
            self.__engagement += 1
        else:
            self.__non_engagement += 1

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
        return self.__engagement > self.__non_engagement

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
        return self.__engagement > self.__non_engagement

    @property
    def engagement(self) -> int:
        """Get the number of learner's engagement on the training data.

        Returns
        -------
        int
            The number of learner's engagement on the training data.

        """
        return self.__engagement

    @property
    def non_engagement(self) -> int:
        """Get the number of learner's non-engagement on the training data.

        Returns
        -------
        int
            The number of learner's non-engagement on the training data.

        """
        return self.__non_engagement
