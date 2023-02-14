from typing import Any, Dict
from typing_extensions import Self

from truelearn.models import EventModel
from ._base import BaseClassifier


class MajorityClassifier(BaseClassifier):
    """A classifier that predicts based on \
    the number of learner's engagement and non-engagement.

    If the number of engagement on the training data is greater than
    the number of non-engagement, the classifier predicts Engage (True);
    otherwise, it predicts Non-Engage (False).
    """

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "engagement": int,
        "non_engagement": int,
    }

    def __init__(self, *, engagement: int = 0, non_engagement: int = 0) -> None:
        """Init MajorityClassifier object.

        Args:
            *: Use to reject positional arguments.
            engagement: The number of learner's engagement.
            non_engagement: The number of learner's non_engagement.
        """
        self._validate_params(
            engagement=engagement,
            non_engagement=non_engagement,
        )

        super().__init__()

        self.engagement = engagement
        self.non_engagement = non_engagement

    def fit(self, _x: EventModel, y: bool) -> Self:
        if y:
            self.engagement += 1
        else:
            self.non_engagement += 1

        return self

    def predict(self, _x: EventModel) -> bool:
        return self.engagement > self.non_engagement

    def predict_proba(self, _x: EventModel) -> float:
        return self.engagement > self.non_engagement
