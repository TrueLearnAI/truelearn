from typing import Any, Dict
from typing_extensions import Self

from truelearn.models import EventModel
from ._base import BaseClassifier


class PersistentClassifier(BaseClassifier):
    """A classifier that makes predictions based on \
    whether the learner has engaged with the last learnable unit.

    Examples:
        >>> from truelearn.learning import PersistentClassifier
        >>> from truelearn.models import EventModel
        >>> persistent = PersistentClassifier()
        >>> persistent
        PersistentClassifier()
        >>> # prepare event model with empty knowledge
        >>> events = [EventModel(), EventModel(), EventModel()]
        >>> engage_stats = [False, True, False]
        >>> for event, engage_stats in zip(events, engage_stats):
        ...     persistent = persistent.fit(event, engage_stats)
        ...     print(persistent.predict(event))
        ...
        False
        True
        False
    """

    _parameter_constraints: Dict[str, Any] = {
        **BaseClassifier._parameter_constraints,
        "engage_with_last": bool,
    }

    def __init__(self, engage_with_last: bool = False) -> None:
        """Init PersistentClassifier object.

        Args:
            engage_with_last: whether the learner engages with the last learnable unit.
        """
        self._validate_params(engage_with_last=engage_with_last)

        super().__init__()

        self.engage_with_last = engage_with_last

    def fit(self, _x: EventModel, y: bool) -> Self:
        self.engage_with_last = y
        return self

    def predict(self, _x: EventModel) -> bool:
        return self.engage_with_last

    def predict_proba(self, _x: EventModel) -> float:
        return float(self.engage_with_last)
