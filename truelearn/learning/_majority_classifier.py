from __future__ import annotations


class MajorityClassifier:
    """A Classifier that predicts based on the number of learner's engagement and non-engagement.

    If the number of engagement on the training data is greater than the number of non-engagement,
    the classifier predicts Engage (True); otherwise, it predicts Non-Engage (False).

    Methods
    -------
    fit(x, y)
        Train the model based on the given data and label.
    predict(x)
        Predict whether the learner will engage.
    predict_prob(x)
        Predict the probability of learner engagement.

    Properties
    ----------
    engagement
    non_engagement

    """

    def __init__(self) -> None:
        self._engagement = 0
        self._non_engagement = 0

    # TODO: add type annotations
    def fit(self, _x, y) -> MajorityClassifier:
        """Train the model based on the given data and labels.

        Parameters
        ----------
        _x: an iterable collection of Topic
            An array of Topic that represents a learnable unit.
        y: bool
            A label that is either True or False.

        Returns
        -------
        The updated model.

        Notes
        -----
        Given the nature of this classifier, the input _x is not used.

        """
        if y:
            self._engagement += 1
        else:
            self._non_engagement += 1

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
        return self._engagement > self._non_engagement

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
        return self._engagement > self._non_engagement

    @property
    def engagement(self) -> int:
        """Get the number of learner's engagement on the training data.

        Returns
        -------
        int
            The number of learner's engagement on the training data.

        """
        return self._engagement

    @property
    def non_engagement(self) -> int:
        """Get the number of learner's non-engagement on the training data.

        Returns
        -------
        int
            The number of learner's non-engagement on the training data.

        """
        return self._non_engagement
