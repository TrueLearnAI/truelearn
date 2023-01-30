class Topic:
    """A class that represents a Topic.

    Parameters
    ----------
    topic_id: a hashable and comparable object
        An id that uniquely identify a Topic.
    description: str or None
        An optional description of the Topic.

    Attributes
    ----------
    topic_id
    description

    """

    def __init__(self, topic_id, description: str | None = None) -> None:
        self.__topic_id = topic_id
        self.__description = description

    def __hash__(self) -> int:
        """Generate the hash of the Topic object.

        Returns
        -------
        int
            Hash of the Topic object. The hash is based on the given topic_id.

        """
        return hash(self.__topic_id)

    def __eq__(self, other) -> bool:
        """Check whether the Topic object equals to another object.

        Parameters
        ----------
        other : object

        Returns
        -------
        bool
            Whether the two given object is equal. Two objects (Topic and the other) are equal iff they are both Topic objects and share the same topic_id.

        """
        if isinstance(other, Topic) is not True:
            return False
        return self.__topic_id == other.topic_id

    @property
    def topic_id(self):
        """Return the Topic id.

        Returns
        -------
        topic_id
            A hashable and comparable type.

        """
        return self.__topic_id

    @property
    def description(self) -> str | None:
        """Return the description of the Topic.

        Returns
        -------
        str | None
            An optional description about the topic.

        """
        return self.__description
