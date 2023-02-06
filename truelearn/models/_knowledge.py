from typing import Iterable

from truelearn.models._topic import Topic


class KnowledgeComponent:
    """A class that represents a knowledge component (KC).

    The class can be used as a founding block to build up knowledge.

    Parameters
    ----------
    topic: Topic
    mean: float
    variance: float

    Attributes
    ----------
    topic
    mean
    variance

    """

    def __init__(self, topic: Topic, mean: float, variance: float) -> None:
        self.__topic = topic
        self.__mean = mean
        self.__variance = variance

    @property
    def topic(self) -> Topic:
        """Return the topic associated with this KC.

        Returns
        -------
        Topic

        """
        return self.__topic

    @property
    def mean(self) -> float:
        """Return the mean associated with this KC.

        Returns
        -------
        float

        """
        return self.__mean

    @property
    def variance(self) -> float:
        """Return the variance associated with this KC.

        Returns
        -------
        float

        """
        return self.__variance

    @mean.setter
    def mean(self, mean: float) -> None:
        """Update the mean associated with this KC.

        Parameters
        ----------
        mean : float
            The new mean value.

        """
        self.__mean = mean

    @variance.setter
    def variance(self, variance: float) -> None:
        """Update the variance with this KC.

        Parameters
        ----------
        variance : float
            The new variance value.

        """
        self.__variance = variance


class Knowledge:
    """A class that represents the knowledge.

    The class can be used to represent 1) the learner's knowledge and
    2) the topics in a learnable unit and the depth of knowledge of those topics.

    Parameters
    ----------
    knowledge: dict[int, KnowledgeComponent]

    Methods
    -------
    get(topic_id, default)
        Get the KC associated with the topic_id from Knowledge. If the topic_id is not included in learner's knowledge,
        the default is returned.
    update(topic, mean, variance)
        Update the mean and variance of the given topic.

    Attributes
    ----------
    topics
    means
    variances

    """

    def __init__(self, knowledge: dict[int, KnowledgeComponent] | None = None) -> None:
        if knowledge is not None:
            self.__knowledge = knowledge
        else:
            self.__knowledge: dict[int, KnowledgeComponent] = {}

    def get(self, topic_id: int, default: KnowledgeComponent | None = None):
        """Get the KnowledgeComponent associated with the topic_id if the topic is in the Knowledge, else default.

        Parameters
        ----------
        topic_id : int
            The id that uniquely identifies a topic.
        default : KnowledgeComponent | None, optional
            The default knowledge component to return when there is no matching KC

        Returns
        -------
        KnowledgeComponent | None

        """
        return self.__knowledge.get(topic_id, default)

    def update(self, topic_id: int, kc: KnowledgeComponent, mean: float, variance: float) -> None:
        """Update the mean and variance of KC associated with the topic_id if the topic is in the Knowledge, else add the given KC into Knowledge.

        The new KC will be created from the kc parameter via `KnowledgeComponent(kc.topic, mean, variance)`.

        Parameters
        ----------
        topic_id : int
            The id that uniquely identifies a topic.
        kc: KnowledgeComponent
            The new KC.
        mean : float
            Deprecated -- will be removed in the next commit
        variance : float
            Deprecated -- will be removed in the next commit

        """
        if topic_id not in self.__knowledge:
            self.__knowledge[topic_id] = KnowledgeComponent(
                kc.topic, mean, variance)
        else:
            self.__knowledge[topic_id].mean = mean
            self.__knowledge[topic_id].variance = variance

    def topic_kc_pairs(self) -> Iterable[tuple[int, KnowledgeComponent]]:
        """Return an iterable of the (topic_id, KC) pair.

        Returns
        -------
        Iterable[tuple[int, KnowledgeComponent]]

        """
        return self.__knowledge.items()

    def knowledge_components(self) -> Iterable[KnowledgeComponent]:
        """Return an iterable of the Knowledge Component.

        Returns
        -------
        Iterable[KnowledgeComponent]

        """
        return self.__knowledge.values()
