from typing import Any, Dict
from typing_extensions import Self

from truelearn.models import EventModel
from truelearn.base import BaseClassifier
from .._constraint import TypeConstraint


class MajorityClassifier(BaseClassifier):
    """A classifier that predicts based on \
    the number of learner's engagement and non-engagement.

    If the number of engagements on the training data is greater than
    the number of non-engagement, the classifier predicts Engage (True);
    otherwise, it predicts Non-Engage (False).

    Examples:
        >>> from truelearn.learning import MajorityClassifier
        >>> from truelearn.models import EventModel
        >>> majority = MajorityClassifier()
        >>> majority
        MajorityClassifier()
        >>> # prepare an event model with empty knowledge
        >>> events = [EventModel(), EventModel(), EventModel()]
        >>> engage_stats = [False, True, True]
        >>> for event, engage_stats in zip(events, engage_stats):
        ...     majority = majority.fit(event, engage_stats)
        ...     print(majority.predict(event))
        ...
        False
        False
        True
    """

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "engagement": TypeConstraint(int),
        "non_engagement": TypeConstraint(int),
        "threshold": TypeConstraint(float),
    }

    def __init__(
        self, *, engagement: int = 0, non_engagement: int = 0, threshold: float = 0.5
    ) -> None:
        """Init MajorityClassifier object.

        Args:
            *: Use to reject positional arguments.
            engagement: The number of learner's engagements.
            non_engagement: The number of learner's non_engagements.
            threshold: A float that determines the classification threshold.

        Raises:
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
        """
        super().__init__()

        self._engagement = engagement
        self._non_engagement = non_engagement
        self._threshold = threshold

        self._validate_params()

    def fit(self, x: EventModel, y: bool) -> Self:
        if y:
            self._engagement += 1
        else:
            self._non_engagement += 1

        return self

    def predict(self, x: EventModel) -> bool:
        return self.predict_proba(x) > self._threshold

    def predict_proba(self, x: EventModel) -> float:
        return self._engagement / max(1, self._engagement + self._non_engagement)
