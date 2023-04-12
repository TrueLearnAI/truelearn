from typing import Any, Dict
from typing_extensions import Self

from truelearn.models import EventModel
from truelearn.base import BaseClassifier
from .._constraint import TypeConstraint


class PersistentClassifier(BaseClassifier):
    """A classifier that makes predictions based on \
    whether the learner has engaged with the last learnable unit.

    Examples:
        >>> from truelearn.learning import PersistentClassifier
        >>> from truelearn.models import EventModel
        >>> persistent = PersistentClassifier()
        >>> persistent
        PersistentClassifier()
        >>> # prepare an event model with empty knowledge
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
        "engage_with_last": TypeConstraint(bool),
    }

    def __init__(self, *, engage_with_last: bool = False) -> None:
        """Init PersistentClassifier object.

        Args:
            engage_with_last: whether the learner engages with the last learnable unit.

        Raises:
            TrueLearnTypeError:
                Types of parameters do not satisfy their constraints.
            TrueLearnValueError:
                Values of parameters do not satisfy their constraints.
        """
        super().__init__()

        self._engage_with_last = engage_with_last

        self._validate_params()

    def fit(self, x: EventModel, y: bool) -> Self:
        self._engage_with_last = y
        return self

    def predict(self, x: EventModel) -> bool:
        return self._engage_with_last

    def predict_proba(self, x: EventModel) -> float:
        return float(self._engage_with_last)
