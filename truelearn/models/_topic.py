class Topic:
    """A class that represents a Topic.

    Parameters
    ----------
    title: str or None
        An optional string that is the title of the topic.
    description: str or None
        An optional string that describes the topic.
    url: str or None
        An optional string that is the url of the topic.

    Attributes
    ----------
    title
    description
    url

    """

    def __init__(self, title: str | None = None, description: str | None = None, url: str | None = None) -> None:
        self.__title = title
        self.__description = description
        self.__url = url

    @property
    def title(self):
        """Return the Topic id.

        Returns
        -------
        str | None
            An optional string that is the title of the topic.

        """
        return self.__title

    @property
    def description(self) -> str | None:
        """Return the description of the Topic.

        Returns
        -------
        str | None
            An optional string that is the description of the topic.

        """
        return self.__description

    @property
    def url(self) -> str | None:
        """Return the url of the Topic.

        Returns
        -------
        str | None
            An optional string that is the url of the topic.

        """
        return self.__url
