from ._topic import Topic


class Knowledge:
    """A class that represents the knowledge.

    The class can be used to represent 1) the learner's knowledge and 2) the topics in a learnable unit and the depth of knowledge of those topics.

    Parameters
    ----------
    means: dict[Topic, float]
    variances: dict[Topic, float]

    Methods
    -------
    update(topic, mean, variance)
        Update the mean and variance of the given topic.

    Attributes
    ----------
    topics
    means
    variances

    Notes
    -----
    The given list of topics, means, and variances should have the same size.
    Otherwise, the initialized Knowledge will be empty.

    """

    def __init__(self, means: dict[Topic, float] | None = None, variances: dict[Topic, float] | None = None) -> None:
        if means is None or means is None or variances is None:
            self.__means: dict[Topic, float] = {}
            self.__variances: dict[Topic, float] = {}
        else:
            assert means.keys() == variances.keys()
            self.__means = means
            self.__variances = variances

    def __contains__(self, topic: Topic) -> bool:
        """Check if the topic is included in the Knowledge.

        Parameters
        ----------
        topic : Topic

        Returns
        -------
        bool
            Whether the topic is included in the Knowledge.

        """
        return topic in self.__means

    @property
    def topics(self) -> set[Topic]:
        """Return a set of topics included in the Knowledge

        Returns
        -------
        set[Topic]
            A set of topics included in the knowledge

        """
        return set(self.__means.keys())

    @property
    def means(self) -> dict[Topic, float]:
        return self.__means

    @property
    def variances(self) -> dict[Topic, float]:
        """Return the variances of all the topic

        Returns
        -------
        dict[Topic, float]
            The variances of all the topics

        """
        return self.__variances

    def update(self, topic: Topic, mean: float, variance: float) -> None:
        """Update the mean and variance of the given topic

        Parameters
        ----------
        topic : Topic
        mean : float
        variance : float

        """
        self.__means[topic] = mean
        self.__variances[topic] = variance
