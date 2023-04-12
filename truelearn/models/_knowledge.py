import collections
import itertools
from typing import Iterable, Hashable, Any, Optional, Dict, Tuple, Deque
from typing_extensions import Self

from ._base import BaseKnowledgeComponent
from truelearn.errors import TrueLearnValueError


class KnowledgeComponent(BaseKnowledgeComponent):
    """A concrete class that implements BaseKnowledgeComponent.

    Examples:
        >>> from truelearn.models import KnowledgeComponent
        >>> kc = KnowledgeComponent(mean=0.0, variance=1.0)
        >>> kc
        KnowledgeComponent(mean=0.0, variance=1.0, timestamp=None, title=None, \
description=None, url=None)
        >>> KnowledgeComponent(mean=0.0, variance=1.0, title="Hello World", \
description="First Program")
        KnowledgeComponent(mean=0.0, variance=1.0, timestamp=None, \
title='Hello World', description='First Program', url=None)
        >>> # update the mean and variance
        >>> kc.update(mean=1.0, variance=2.0)
        >>> kc
        KnowledgeComponent(mean=1.0, variance=2.0, timestamp=None, title=None, \
description=None, url=None)
        >>> # clone with new mean and variance
        >>> kc.clone(mean=0.0, variance=1.0)
        KnowledgeComponent(mean=0.0, variance=1.0, timestamp=None, title=None, \
description=None, url=None)
    """

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
                A float
                indicating the POSIX timestamp of when the knowledge component \
                was last updated.
            title:
                An optional string
                storing the title of the knowledge component.
            description:
                An optional string that describes the knowledge component.
            url:
                An optional string
                storing the url of the knowledge component.

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
            f"KnowledgeComponent(mean={self.mean!r}, variance={self.variance!r}, "
            f"timestamp={self.timestamp!r}, title={self.title!r}, "
            f"description={self.description!r}, url={self.url!r})"
        )

    @property
    def title(self) -> Optional[str]:
        """Optional[str]: The title of the knowledge component."""
        return self.__title

    @property
    def description(self) -> Optional[str]:
        """Optional[str]: The description of the knowledge component."""
        return self.__description

    @property
    def url(self) -> Optional[str]:
        """Optional[str]: The url of the knowledge component."""
        return self.__url

    @property
    def mean(self) -> float:
        """float: The mean of the knowledge component."""
        return self.__mean

    @property
    def variance(self) -> float:
        """float: The variance of the knowledge component."""
        return self.__variance

    @property
    def timestamp(self) -> Optional[float]:
        """Optional[float]: The POSIX timestamp of when the knowledge component was \
        last updated."""
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
                The new mean of the KnowledgeComponent.
            variance:
                The new variance of the KnowledgeComponent.
            timestamp:
                An optional new POSIX timestamp of the KnowledgeComponent.
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
    """A knowledge component that keeps a history about how it was updated.

    Examples:
        >>> from truelearn.models import HistoryAwareKnowledgeComponent
        >>> hkc = HistoryAwareKnowledgeComponent(mean=0.0, variance=1.0)
        >>> hkc
        HistoryAwareKnowledgeComponent(mean=0.0, variance=1.0, timestamp=None, \
title=None, description=None, url=None, history=deque([], maxlen=None))
        >>> # update the mean and variance of the hkc
        >>> hkc.update(mean=1.0, variance=2.0)
        >>> hkc
        HistoryAwareKnowledgeComponent(mean=1.0, variance=2.0, timestamp=None, \
title=None, description=None, url=None, history=deque([(0.0, 1.0, None)], maxlen=None))
        >>> # clone the history aware knowledge component with given mean and variance
        >>> hkc.clone(mean=2.0, variance=3.0)
        HistoryAwareKnowledgeComponent(mean=2.0, variance=3.0, timestamp=None, \
title=None, description=None, url=None, history=deque([(0.0, 1.0, None)], maxlen=None))
    """

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

        history = collections.deque(history or [], maxlen=history_limit)
        self.__history = history

    def __repr__(self, n_max_object: int = 1) -> str:
        """Get a description of the history aware classifier object.

        Args:
            n_max_object:
                A positive int specifying the maximum number of
                history records to be printed.

        Returns:
            A string description of the HistoryAwareKnowledgeComponent object.

        Raises:
            TrueLearnValueError:
                If the n_max_object is less than 0.
        """
        if n_max_object < 0:
            raise TrueLearnValueError(
                f"Expected n_max_object>=0. Got n_max_object={n_max_object} instead."
            )

        printed_number = min(len(self.history), n_max_object)
        history_to_format = map(repr, itertools.islice(self.history, printed_number))

        if len(self.history) == printed_number:
            history_fmt_str = f"[{', '.join(history_to_format)}]"
        else:
            history_fmt_str = f"[{', '.join(history_to_format)}, ...]"

        return (
            f"HistoryAwareKnowledgeComponent(mean={self.mean!r}, "
            f"variance={self.variance!r}, "
            f"timestamp={self.timestamp!r}, title={self.title!r}, "
            f"description={self.description!r}, url={self.url!r}, "
            # history_fmt_str is a deque formatted to string
            # we don't need to use !r to add quotes around it
            f"history=deque({history_fmt_str}, maxlen={self.history.maxlen!r}))"
        )

    @property
    def history(self) -> Deque[Tuple[float, float, Optional[float]]]:
        """Deque[Tuple[float, float, Optional[float]]]: The update history of the \
        current knowledge component."""
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
            title=self.title,
            description=self.description,
            url=self.url,
            history=self.__history,
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

    Examples:
        >>> from truelearn.models import Knowledge, KnowledgeComponent
        >>> # construct empty knowledge
        >>> knowledge = Knowledge()
        >>> knowledge
        Knowledge(knowledge={})
        >>> # add a new kc to knowledge
        >>> knowledge.update_kc(1, KnowledgeComponent(mean=1.0, variance=2.0))
        >>> knowledge
        Knowledge(knowledge={1: KnowledgeComponent(mean=1.0, variance=2.0, \
timestamp=None, title=None, description=None, url=None)})
        >>> # get a kc from the knowledge
        >>> knowledge.get_kc(1, KnowledgeComponent(mean=0.0, variance=1.0))
        KnowledgeComponent(mean=1.0, variance=2.0, timestamp=None, title=None, \
description=None, url=None)
        >>> # get iterator of keys and (key, value) pairs
        >>> knowledge.knowledge_components()
        dict_values([KnowledgeComponent(mean=1.0, variance=2.0, timestamp=None, \
title=None, description=None, url=None)])
        >>> knowledge.topic_kc_pairs()
        dict_items([(1, KnowledgeComponent(mean=1.0, variance=2.0, timestamp=None, \
title=None, description=None, url=None))])
    """

    def __init__(
        self,
        knowledge: Optional[Dict[Hashable, BaseKnowledgeComponent]] = None,
    ) -> None:
        """Init the Knowledge object.

        If the given knowledge is None, the knowledge will be initialized emptily.

        Args:
            knowledge: A dict mapping a hashable id of the knowledge component
                to the corresponding knowledge component object.
        """
        super().__init__()

        self.__knowledge = knowledge or {}

    def __repr__(self, n_max_object: int = 1) -> str:
        """Print the Knowledge object.

        Args:
            n_max_object:
                A positive int specifying the maximum number of
                knowledge components to be printed.

        Returns:
            A string description of the Knowledge object.

        Raises:
            TrueLearnValueError:
                If the n_max_object is less than 0.
        """
        if n_max_object < 0:
            raise TrueLearnValueError(
                f"Expected n_max_object>=0. Got n_max_object={n_max_object} instead."
            )

        printed_number = min(len(self.__knowledge), n_max_object)
        kc_to_format = map(
            lambda kv_pair: f"{kv_pair[0]!r}: {kv_pair[1]!r}",
            itertools.islice(self.__knowledge.items(), printed_number),
        )

        if len(self.__knowledge) == printed_number:
            knowledge_fmt_str = f"{{{', '.join(kc_to_format)}}}"
        else:
            knowledge_fmt_str = f"{{{', '.join(kc_to_format)}, ...}}"

        # don't need to add !r as we format a dict to str
        return f"Knowledge(knowledge={knowledge_fmt_str})"

    def get_kc(
        self, topic_id: Hashable, default: BaseKnowledgeComponent
    ) -> BaseKnowledgeComponent:
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

    def update_kc(self, topic_id: Hashable, kc: BaseKnowledgeComponent) -> None:
        """Update the knowledge component associated with the given topic_id.

        If the topic_id doesn't exist in the AbstractKnowledge, the mapping
        from the topic_id to the knowledge component will be created.

        Args:
          topic_id: Hashable:
          kc: BaseKnowledgeComponent:
        """
        self.__knowledge[topic_id] = kc

    def topic_kc_pairs(
        self,
    ) -> Iterable[Tuple[Hashable, BaseKnowledgeComponent]]:
        """Return an iterable of the (topic_id, knowledge_component) pair."""
        return self.__knowledge.items()

    def knowledge_components(self) -> Iterable[BaseKnowledgeComponent]:
        """Return an iterable of the knowledge component."""
        return self.__knowledge.values()
