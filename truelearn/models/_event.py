import dataclasses
from typing import Optional

from ._knowledge import Knowledge


@dataclasses.dataclass
class EventModel:
    """A class that models a learning event in TrueLearn algorithm.

    Examples:
        >>> from truelearn.models import EventModel, KnowledgeComponent
        >>> # construct an empty event model
        >>> EventModel()
        EventModel(knowledge=Knowledge(knowledge={}), event_time=None)
        >>> # construct an event model with given event_time
        >>> EventModel(event_time=0.0)
        EventModel(knowledge=Knowledge(knowledge={}), event_time=0.0)
        >>> # construct an event model with given knowledge
        >>> knowledge = Knowledge({1: KnowledgeComponent(mean=0.0, variance=1.0)})
        >>> EventModel(knowledge=knowledge)  # doctest:+ELLIPSIS
        EventModel(knowledge=Knowledge(knowledge={1: KnowledgeComponent(mean=0.0, \
variance=1.0, ...)}), event_time=None)
    """

    knowledge: Knowledge = dataclasses.field(default_factory=Knowledge)
    """A knowledge representation of the educational resources."""

    event_time: Optional[float] = None
    """A float that specifies the POSIX timestamp when the event occurs."""
