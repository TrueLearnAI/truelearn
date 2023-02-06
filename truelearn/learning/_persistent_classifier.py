from __future__ import annotations


class PersistentClassifier:
    """A classifier that makes predictions based on whether the learner has engaged with the last learnable unit.

    Methods
    -------
    fit(x, y)
        Train the model based on the given data and label.
    predict(x)
        Predict whether the learner will engage.
    predict_proba(x)
        Predict the probability of learner engagement.

    Properties
    ----------
    engage_with_last

    """

    def __init__(self) -> None:
        self.__engage_with_last = False

    def fit(self, _x, y) -> PersistentClassifier:
        """Train the model based on the given data and labels.

        Parameters
        ----------
        _x: AbstractKnowledge
            A knowledge representation of a learnable unit.
        y: bool
            A label that is either True or False.

        Returns
        -------
        PersistentClassifier
            The updated model.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        self.__engage_with_last = y
        return self

    def predict(self, _x) -> bool:
        """Predict whether the learner will engage.

        Parameters
        ----------
        _x: AbstractKnowledge
            A knowledge representation of a learnable unit.

        Returns
        -------
        bool
            Whether the learner will engage with the given learnable unit.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        return self.__engage_with_last

    def predict_proba(self, _x) -> float:
        """Predict the probability of learner engagement.

        Parameters
        ----------
        _x: AbstractKnowledge
            A knowledge representation of a learnable unit.

        Returns
        -------
        float
            The probability that the learner will engage with the given learnable unit.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        return float(self.__engage_with_last)

    @property
    def engage_with_last(self) -> bool:
        """Return whether the learner engage with the last learnable unit.

        Returns
        -------
        bool
            Whether the learner engage with the last learnable unit.

        """
        return self.__engage_with_last
