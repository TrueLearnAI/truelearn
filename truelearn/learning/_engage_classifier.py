from __future__ import annotations


class EngageClassifier:
    """A Classifier that always makes positive prediction.

    Methods
    -------
    fit(x, y)
        Train the model based on the given data and label.
    predict(x)
        Predict whether the learner will engage.
    predict_prob(x)
        Predict the probability of learner engagement.

    """

    def __init__(self) -> None:
        pass

    # TODO: add type annotations
    def fit(self, _x, _y) -> EngageClassifier:
        """Train the model based on the given data and labels.

        Parameters
        ----------
        _x: an iterable collection of Topic
            An array of Topic that represents a learnable unit.
        _y: bool
            A label that is either True or False.

        Returns
        -------
        The updated model.

        Notes
        -----
        Given the nature of this classifier, the input _x and _y are not used.

        """
        return self

    # TODO: add type annotations
    def predict(self, _x) -> bool:
        """Predict whether the learner will engage.

        Parameters
        ----------
        _x: an iterable collection of Topic
            An array of Topic that represents a learnable unit.

        Returns
        -------
        bool
            Whether the learner will engage with the given learnable unit.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        return True

    # TODO: add type annotations
    def predict_prob(self, _x) -> float:
        """Predict the probability of learner engagement.

        Parameters
        ----------
        _x: an iterable collection of Topic
            An array of Topic that represents a learnable unit.

        Returns
        -------
        float
            The probability that the learner will engage with the given learnable unit.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        return 1.
