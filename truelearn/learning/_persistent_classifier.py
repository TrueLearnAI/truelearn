from typing import Any, Dict
from typing_extensions import Self

from ._base import BaseClassifier
from truelearn.models import EventModel


class PersistentClassifier(BaseClassifier):
    """A classifier that makes predictions based on \
    whether the learner has engaged with the last learnable unit."""

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "engage_with_last": bool,
    }

    def __init__(self, engage_with_last: bool = False) -> None:
        """Init PersistentClassifier object.

        Args:
            engage_with_last: whether the learner engages with the last learnable unit.
        """
        self._validate_params(
            engage_with_last=engage_with_last
        )

        super().__init__()

        self.engage_with_last = engage_with_last

    def fit(self, _x: EventModel, y: bool) -> Self:
        self.engage_with_last = y
        return self

    def predict(self, _x: EventModel) -> bool:
        return self.engage_with_last

    def predict_proba(self, _x: EventModel) -> float:
        return float(self.engage_with_last)
