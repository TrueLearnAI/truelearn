from typing_extensions import Self, Any

from ._base import BaseClassifier
from truelearn.models import EventModel


class EngageClassifier(BaseClassifier):
    """A Classifier that always makes positive prediction."""

    _parameter_constraints: dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
    }

    def __init__(self) -> None:
        """Init EngageClassifier object."""
        super().__init__()

        self._validate_params()

    def fit(self, _x: EventModel, _y: bool) -> Self:
        return self

    def predict(self, _x: EventModel) -> bool:
        return True

    def predict_proba(self, _x: EventModel) -> float:
        return 1.0
