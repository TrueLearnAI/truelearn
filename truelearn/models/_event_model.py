import dataclasses

from ._knowledge import Knowledge


@dataclasses.dataclass
class EventModel:
    """A class that models a learning event in TrueLearn algorithm.

    Parameters
    ----------
    knowledge: Knowledge | None
        The knowledge representation of the learnable unit.
        A new Knowledge will be created if the knowledge is None.
    event_time: float | None, optional
        The POSIX timestamp when the learning event happens.
        If the timestamp is None, the event timestamp will be None,
        which affects the training of classifiers that utilize Interest.

    Properties
    ----------
    knowledge
    event_time

    # TODO: rename Properties to Attributes after switching to google style

    """

    knowledge: Knowledge = dataclasses.field(default_factory=Knowledge)
    event_time: float | None = None
