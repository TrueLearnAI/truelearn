from typing import Dict, Any
from typing_extensions import Self

from truelearn.models import EventModel
from truelearn.base import BaseClassifier


class EngageClassifier(BaseClassifier):
    """A Classifier that always makes a positive prediction.

    Examples:
        >>> from truelearn.learning import EngageClassifier
        >>> from truelearn.models import EventModel
        >>> engage = EngageClassifier()
        >>> engage
        EngageClassifier()
        >>> # prepare an event model with empty knowledge
        >>> event = EventModel()
        >>> engage.fit(event, False)
        EngageClassifier()
        >>> engage.predict(event)
        True
        >>> engage.predict_proba(event)
        1.0
    """

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
    }

    def __init__(self) -> None:
        """Init EngageClassifier object."""
        super().__init__()

        self._validate_params()

    def fit(self, x: EventModel, y: bool) -> Self:
        return self

    def predict(self, x: EventModel) -> bool:
        return True

    def predict_proba(self, x: EventModel) -> float:
        return 1.0
