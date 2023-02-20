import collections
from typing import Iterable, Hashable, Any, Optional, Dict, Tuple, Deque
from typing_extensions import Self

from ._abstract_knowledge import AbstractKnowledgeComponent


class KnowledgeComponent(AbstractKnowledgeComponent):
    """A concrete class that implements AbstractKnowledgeComponent."""

    def __init__(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: Optional[float] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        """Init the KnowledgeComponent object.

        Args:
            mean:
                A float indicating the mean of the knowledge component.
            variance:
                A float indicating the variance of the knowledge component.
            timestamp:
                A float indicating the POSIX timestamp of the last update of
                the knowledge component.
            title:
                An optional string storing the title of the knowledge component.
            description:
                An optional string that describes the knowledge component.
            url:
                An optional string storing the url of the knowledge component.

        Returns:
            None.
        """
        super().__init__()

        self.__mean = mean
        self.__variance = variance
        self.__timestamp = timestamp

        self.__title = title
        self.__description = description
        self.__url = url

    def __repr__(self) -> str:
        """Get a description of the knowledge component object.

        Returns:
            A string description of the KnowledgeComponent object.
            It prints all the attributes of this object.
        """
        return (
            f"KnowledgeComponent(mean={self.mean}, variance={self.variance}, "
            f"timestamp={self.timestamp}, title={self.title}, "
            f"description={self.description}, url={self.url})"
        )

    @property
    def title(self) -> Optional[str]:
        """The title of the knowledge component."""
        return self.__title

    @property
    def description(self) -> Optional[str]:
        """The description of the knowledge component."""
        return self.__description

    @property
    def url(self) -> Optional[str]:
        """The url of the knowledge component."""
        return self.__url

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def variance(self) -> float:
        return self.__variance

    @property
    def timestamp(self) -> Optional[float]:
        return self.__timestamp

    def update(
        self,
        *,
        mean: Optional[float] = None,
        variance: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        if mean is not None:
            self.__mean = mean
        if variance is not None:
            self.__variance = variance
        if timestamp is not None:
            self.__timestamp = timestamp

    def clone(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: Optional[float] = None,
    ) -> Self:
        """Generate a copy of the current knowledge component with \
        given mean, variance and timestamp.

        Args:
            *:
                Use to reject positional arguments.
            mean:
                The new mean of the AbstractKnowledgeComponent.
            variance:
                The new variance of the AbstractKnowledgeComponent.
            timestamp:
                An optional new POSIX timestamp of the AbstractKnowledgeComponent.
                If None is given, the timestamp of the cloned knowledge component is
                assigned to None.

        Returns:
            A cloned knowledge component with given mean, variance and timestamp.

        """
        return KnowledgeComponent(
            mean=mean,
            variance=variance,
            timestamp=timestamp,
            title=self.__title,
            description=self.__description,
            url=self.__url,
        )

    def export_as_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.__mean,
            "variance": self.__variance,
            "timestamp": self.__timestamp,
            "title": self.__title,
            "description": self.__description,
            "url": self.__url,
        }


class HistoryAwareKnowledgeComponent(KnowledgeComponent):
    """A knowledge component that keeps a history about how it was updated."""

    def __init__(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: Optional[float] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        history_limit: Optional[int] = None,
        history: Optional[Deque[Tuple[float, float, Optional[float]]]] = None,
    ) -> None:
        """Init the KnowledgeComponent object.

        Args:
            mean:
                A float indicating the mean of the knowledge component.
            variance:
                A float indicating the variance of the knowledge component.
            timestamp:
                A float indicating the POSIX timestamp of the last update of
                the knowledge component.
            title:
                An optional string storing the title of the knowledge component.
            description:
                An optional string that describes the knowledge component.
            url:
                An optional string storing the url of the knowledge component.
            history_limit:
                A positive int that specifies the number of entries stored in the
                history. If the limit is None, it means there is no limit.
                Defaults to None.
            history:
                A queue that stores the update history of the knowledge component.
                Each entry in the queue is a tuple (mean, variance, timestamp)
                which records the mean and variance of this knowledge component
                at the given timestamp.

        Returns:
            None.
        """
        super().__init__(
            mean=mean,
            variance=variance,
            timestamp=timestamp,
            title=title,
            description=description,
            url=url,
        )

        if history is None:
            history = collections.deque(maxlen=history_limit)
        else:
            if history.maxlen != history_limit:
                history = collections.deque(history, maxlen=history_limit)
        self.__history = history

    def __repr__(self, N_MAX_OBJ: int = 1) -> str:
        """Get a description of the history aware classifier object.

        Args:
            N_MAX_OBJ:
                An int specifying the maximum number of
                history records to be printed.
                Defaults to 1.

        Returns:
            A string description of the HistoryAwareKnowledgeComponent object.
        """
        history_to_format = []
        printed_number = min(len(self.history), N_MAX_OBJ)
        for i in range(printed_number):
            history_to_format.append(repr(self.history[i]))
        if len(self.history) == printed_number:
            history_fmt_str = f"[{','.join(history_to_format)}]"
        else:
            history_fmt_str = f"[{','.join(history_to_format)}, ...]"

        history_maxlen_fmt_str = "inf"
        if self.history.maxlen is not None:
            history_maxlen_fmt_str = str(self.history.maxlen)

        return (
            f"HistoryAwareKnowledgeComponent(mean={self.mean}, "
            f"variance={self.variance}, "
            f"timestamp={self.timestamp}, title={self.title}, "
            f"description={self.description}, url={self.url}, "
            f"history=deque({history_fmt_str}, maxlen={history_maxlen_fmt_str}))"
        )

    @property
    def history(self) -> Deque[Tuple[float, float, Optional[float]]]:
        """The update history of the current knowledge component."""
        return self.__history

    def update(
        self,
        *,
        mean: Optional[float] = None,
        variance: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self.__history.append((self.mean, self.variance, self.timestamp))

        super().update(mean=mean, variance=variance, timestamp=timestamp)

    def clone(
        self,
        *,
        mean: float,
        variance: float,
        timestamp: Optional[float] = None,
    ) -> Self:
        return HistoryAwareKnowledgeComponent(
            mean=mean,
            variance=variance,
            timestamp=timestamp,
            title=self.__title,
            description=self.__description,
            url=self.__url,
            history=self.__history.copy(),
            history_limit=self.__history.maxlen,
        )

    def export_as_dict(self) -> Dict[str, Any]:
        return {**super().export_as_dict(), "history": self.__history}


class Knowledge:
    """The representation of the learner's knowledge.

    In TrueLearn, we assume every learner's knowledge consists of many different
    knowledge components. The Knowledge class is used to represent this relationship.

    The class can be used to represent 1) the learner's knowledge and
    2) the knowledge of a learnable unit.
    """

    def __init__(
        self,
        knowledge: Optional[Dict[Hashable, AbstractKnowledgeComponent]] = None,
    ) -> None:
        """Init the Knowledge object.

        If the given knowledge is None, the knowledge will be initialized emptily.

        Args:
            knowledge: A dict mapping a hashable id of the knowledge component
                to the corresponding knowledge component object. Defaults to None.
        """
        super().__init__()

        if knowledge is None:
            knowledge = {}
        self.__knowledge = knowledge

    def __repr__(self, N_MAX_OBJ: int = 1) -> str:
        """Print the Knowledge object.

        Args:
            N_MAX_OBJ:
                An int specifying the maximum number of
                knowledge components to be printed.
                Defaults to 1.

        Returns:
            A string description of the Knowledge object.
        """
        kc_to_format = []
        printed_number = min(len(self.__knowledge), N_MAX_OBJ)
        for idx, (key, value) in enumerate(self.__knowledge.items()):
            if idx >= printed_number:
                break
            kc_to_format.append(f"{key!r}: {value!r}")

        if len(self.__knowledge) == printed_number:
            knowledge_fmt_str = f"{{{','.join(kc_to_format)}}}"
        else:
            knowledge_fmt_str = f"{{{','.join(kc_to_format)}, ...}}"

        return f"Knowledge({knowledge_fmt_str})"

    def get_kc(
        self, topic_id: Hashable, default: AbstractKnowledgeComponent
    ) -> AbstractKnowledgeComponent:
        """Get the knowledge component associated with the given id.

        Args:
            topic_id: The id that uniquely identifies a knowledge component.
            default: The default knowledge component to return.

        Returns:
            The knowledge component extracted from the knowledge
            if the topic_id exists in the knowledge.
            Otherwise, the default value will be returned.
        """
        return self.__knowledge.get(topic_id, default)

    def update_kc(self, topic_id: Hashable, kc: AbstractKnowledgeComponent) -> None:
        """Update the knowledge component associated with the given topic_id.

        If the topic_id doesn't exist in the AbstractKnowledge, the mapping
        from the topic_id to the knowledge component will be created.

        Args:
          topic_id: Hashable:
          kc: AbstractKnowledgeComponent:
        """
        self.__knowledge[topic_id] = kc

    def topic_kc_pairs(
        self,
    ) -> Iterable[Tuple[Hashable, AbstractKnowledgeComponent]]:
        """Return an iterable of the (topic_id, knowledge_component) pair."""
        return self.__knowledge.items()

    def knowledge_components(self) -> Iterable[AbstractKnowledgeComponent]:
        """Return an iterable of the knowledge component."""
        return self.__knowledge.values()
