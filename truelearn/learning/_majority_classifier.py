from typing_extensions import Self
from typing import Any

from ._base import BaseClassifier
from truelearn.models import EventModel


class MajorityClassifier(BaseClassifier):
    """A classifier that predicts based on the number of learner's engagement and non-engagement.

    If the number of engagement on the training data is greater than the number of non-engagement,
    the classifier predicts Engage (True); otherwise, it predicts Non-Engage (False).
    """

    _parameter_constraints: dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "_engagement": int,
        "_non_engagement": int,
    }

    def __init__(self, engagement: int = 0, non_engagement: int = 0) -> None:
        """Init MajorityClassifier object."""
        super().__init__()

        self._engagement = engagement
        self._non_engagement = non_engagement

        self._validate_params()

    def fit(self, _x: EventModel, y: bool) -> Self:
        if y:
            self._engagement += 1
        else:
            self._non_engagement += 1

        return self

    def predict(self, _x: EventModel) -> bool:
        return self._engagement > self._non_engagement

    def predict_proba(self, _x: EventModel) -> float:
        return self._engagement > self._non_engagement
