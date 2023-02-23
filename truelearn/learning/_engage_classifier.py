from typing import Dict, Any
from typing_extensions import Self

from truelearn.models import EventModel
from ._base import BaseClassifier


class EngageClassifier(BaseClassifier):
    """A Classifier that always makes positive prediction."""

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
    }

    def __init__(self) -> None:
        """Init EngageClassifier object."""
        self._validate_params()

        super().__init__()

    def fit(self, _x: EventModel, _y: bool) -> Self:
        return self

    def predict(self, _x: EventModel) -> bool:
        return True

    def predict_proba(self, _x: EventModel) -> float:
        return 1.0
