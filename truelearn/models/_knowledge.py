from typing import Iterable, Hashable, Any
from typing_extensions import Self

from ._abstract_knowledge import AbstractKnowledgeComponent


class KnowledgeComponent(AbstractKnowledgeComponent):
    """A concrete class that implements AbstractKnowledgeComponent which represents a knowledge component.

    Parameters
    ----------
    mean: float
    variance: float
    timestamp: float | None
    title: str
    description: str
    url: str

    Methods
    -------
    update(mean, variance)
        Update the mean and variance of the KnowledgeComponent
    clone(mean, variance)
        Clone the KnowledgeComponent with new mean and variance
    export(output_format)
        Export the KnowledgeComponent into some format

    Properties
    ----------
    mean
    variance
    timestamp
    title
    description
    url

    """

    def __init__(self,
                 *,
                 mean: float,
                 variance: float,
                 timestamp: float | None = None,
                 title: str | None = None,
                 description: str | None = None,
                 url: str | None = None) -> None:
        super().__init__()

        self.__title = title
        self.__description = description
        self.__url = url

        self.__mean = mean
        self.__variance = variance
        self.__timestamp = timestamp

    @property
    def title(self) -> str | None:
        """Return the title of the KnowledgeComponent.

        Returns
        -------
        str | None
            An optional string that is the title of the KnowledgeComponent.

        """
        return self.__title

    @property
    def description(self) -> str | None:
        """Return the description of the KnowledgeComponent.

        Returns
        -------
        str | None
            An optional string that is the description of the KnowledgeComponent.

        """
        return self.__description

    @property
    def url(self) -> str | None:
        """Return the url of the KnowledgeComponent.

        Returns
        -------
        str | None
            An optional string that is the url of the KnowledgeComponent.

        """
        return self.__url

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def variance(self) -> float:
        return self.__variance

    @property
    def timestamp(self) -> float | None:
        return self.__timestamp

    def update(self,
               *,
               mean: float | None = None,
               variance: float | None = None,
               timestamp: float | None = None) -> None:
        if mean is not None:
            self.__mean = mean
        if variance is not None:
            self.__variance = variance
        if timestamp is not None:
            self.__timestamp = timestamp

    def clone(self,
              *,
              mean: float | None = None,
              variance: float | None = None,
              timestamp: float | None = None) -> Self:
        if mean is None:
            mean = self.__mean
        if variance is None:
            variance = self.__variance
        return KnowledgeComponent(mean=mean, variance=variance, timestamp=timestamp, title=self.__title,
                                  description=self.__description, url=self.__url)

    def export(self, output_format: str) -> Any:
        raise NotImplementedError(
            f"The export function for {output_format} is not yet implemented.")


class Knowledge:
    """An abstract class that represents the knowledge.

    The class can be used to represent 1) the learner's knowledge and
    2) the topics in a learnable unit and the depth of knowledge of those topics.

    Methods
    -------
    get(topic_id, default)
        Get the AbstractKnowledgeComponent associated with the topic_id.
        If the topic_id is not included in learner's knowledge, the default is returned.
    update(topic_id, kc)
        Update the AbstractKnowledgeComponent associated with the topic_id
    topic_kc_pairs()
        Return an iterable of (topic_id, AbstractKnowledgeComponent) pairs.
    knowledge_components()
        Return an iterable of AbstractKnowledgeComponents.

    """

    def __init__(self, knowledge: dict[Hashable, AbstractKnowledgeComponent] | None = None) -> None:
        super().__init__()

        if knowledge is not None:
            self.__knowledge = knowledge
        else:
            self.__knowledge: dict[Hashable, AbstractKnowledgeComponent] = {}

    def get_kc(self, topic_id: Hashable, default: AbstractKnowledgeComponent) -> AbstractKnowledgeComponent:
        """Get the AbstractKnowledgeComponent associated with the topic_id if the AbstractKnowledgeComponent is in\
        the AbstractKnowledge, else return default.

        Parameters
        ----------
        topic_id : Hashable
            The id that uniquely identifies a topic.
        default : AbstractKnowledgeComponent
            The default AbstractKnowledgeComponent to return

        Returns
        -------
        AbstractKnowledgeComponent

        """
        return self.__knowledge.get(topic_id, default)

    def update_kc(self, topic_id: Hashable, kc: AbstractKnowledgeComponent) -> None:
        """Update the AbstractKnowledgeComponent associated with the topic_id.

        If the topic_id doesn't exist in the AbstractKnowledge, the mapping will be created.

        Parameters
        ----------
        topic_id : Hashable
            The id that uniquely identifies a topic.
        kc: AbstractKnowledgeComponent
            The new AbstractKnowledgeComponents.

        """
        self.__knowledge[topic_id] = kc

    def topic_kc_pairs(self) -> Iterable[tuple[Hashable, AbstractKnowledgeComponent]]:
        """Return an iterable of the (topic_id, AbstractKnowledgeComponent) pair.

        Returns
        -------
        Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

        """
        return self.__knowledge.items()

    def knowledge_components(self) -> Iterable[AbstractKnowledgeComponent]:
        """Return an iterable of the AbstractKnowledgeComponents.

        Returns
        -------
        Iterable[AbstractKnowledgeComponent]

        """
        return self.__knowledge.values()

    def export(self, output_format: str) -> Any:
        """Export the AbstractKnowledge into some formats.

        Parameters
        ----------
        output_format : str
            The name of the output format

        Returns
        -------
        Any
            The requested format

        Raises
        ------
        NotImplementedError
            If the requested format is not available

        """
        raise NotImplementedError(
            f"The export function for {output_format} is not yet implemented.")
