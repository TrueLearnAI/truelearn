from __future__ import annotations
from typing import Iterable, Hashable, Any
from typing_extensions import Self
from abc import ABC, abstractmethod


class AbstractKnowledgeComponent(ABC):
    """An abstract class that represents a knowledge component (KC).

    Methods
    -------
    update(mean, variance, timestamp)
        Update the mean and variance of the AbstractKnowledgeComponent
    clone(mean, variance, timestamp)
        Clone the AbstractKnowledgeComponent with new mean and variance
    export(output_format)
        Export the AbstractKnowledgeComponent into some format

    Properties
    ----------
    mean
    variance
    timestamp

    """

    @property
    @abstractmethod
    def mean(self) -> float:
        """Return the mean of this AbstractKnowledgeComponent.

        Returns
        -------
        float

        """

    @property
    @abstractmethod
    def variance(self) -> float:
        """Return the variance of this AbstractKnowledgeComponent.

        Returns
        -------
        float

        """

    @property
    @abstractmethod
    def timestamp(self) -> float | None:
        """Return the POSIX timestamp of the last update of this AbstractKnowledgeComponent.

        Returns
        -------
        float | None

        """

    @abstractmethod
    def update(self, *, mean: float | None = None, variance: float | None = None, timestamp: float | None = None) -> None:
        """Update the mean, variance, and timestamp of this AbstractKnowledgeComponent.

        Parameters
        ----------
        mean : float | None, optional
            The new mean of the AbstractKnowledgeComponent.
            If the given mean is None, it will not be updated.
        variance : float | None, optional
            The new variance of the AbstractKnowledgeComponent.
            If the given variance is None, it will not be updated.
        timestamp : float | None, optional
            The new POSIX timestamp of the AbstractKnowledgeComponent.
            If the timestamp is None, the timestamp of the AbstractKnowledgeComponent will not be updated.

        """

    @abstractmethod
    def clone(self, *, mean: float | None = None, variance: float | None = None, timestamp: float | None = None) -> Self:
        """Generate a copy of the current AbstractKnowledgeComponent with given mean and variance.

        This function doesn't change the mean and variance of the current AbstractKnowledgeComponent.

        Parameters
        ----------
        mean : float | None, optional
            The new mean of the AbstractKnowledgeComponent.
            If the given mean is None, it will not be updated.
        variance : float | None, optional
            The new variance of the AbstractKnowledgeComponent.
            If the given variance is None, it will not be updated.
        timestamp : float | None, optional
            The new POSIX timestamp of the AbstractKnowledgeComponent.
            If the timestamp is None, the timestamp of the AbstractKnowledgeComponent will not be updated.

        Returns
        -------
        Self
            A cloned version of the current knowledge with the given parameters.

        """

    @abstractmethod
    def export(self, output_format: str) -> Any:
        """Export the AbstractKnowledgeComponent into some formats.

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


class AbstractKnowledge(ABC):
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def topic_kc_pairs(self) -> Iterable[tuple[Hashable, AbstractKnowledgeComponent]]:
        """Return an iterable of the (topic_id, AbstractKnowledgeComponent) pair.

        Returns
        -------
        Iterable[tuple[Hashable, AbstractKnowledgeComponent]]

        """

    @abstractmethod
    def knowledge_components(self) -> Iterable[AbstractKnowledgeComponent]:
        """Return an iterable of the AbstractKnowledgeComponents.

        Returns
        -------
        Iterable[AbstractKnowledgeComponent]

        """

    @abstractmethod
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
