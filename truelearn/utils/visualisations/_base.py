import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Hashable
from typing_extensions import final

import matplotlib.pyplot as plt
from plotly import graph_objects as go

from truelearn.errors import TrueLearnTypeError
from truelearn.models import Knowledge


class BasePlotter(ABC):
    """The base class of all the plotters."""

    @final
    def _standardise_data(
        self,
        raw_data: Knowledge,
        history: bool = False,
        topics: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Convert a Knowledge object to one suitable for generating visualisations.

        Optional utility function that converts the learner's knowledge (obtainable
        via learner.knowledge) to the Iterable used by plot().

        Args:
            raw_data:
                The learner's knowledge, represented by a Knowledge object.
            history:
                Whether the user wants to use the knowledge component's timestamps
                (this allows the visualisations to display more information on
                hover such as the number of videos watched and the last video watched).
                If set to True, the Knowledge object must consist of knowledge
                components that respect HistoryAwareKnowledgeComponents protocol,
                or an error will be raised.
            topics:
                An optional iterable of topics. The method will extract all the topics
                that are present in this iterables.
                If not specified, all topics are extracted.

        Returns:
            A tuple containing (content, rest) where content contains topics that are in
            the iterable of topics and rest contains topics that are not in
            the iterable of topics.
        """
        # make topics a set
        if topics is not None:
            topics = set(topics)

        raw_data_dict = knowledge_to_dict(raw_data)

        content = []
        rest = []

        for kc in raw_data_dict.values():
            data = self._get_kc_details(kc, history)
            if topics is None or data[2] in topics:
                content.append(data)
            else:
                rest.append(data)

        content.sort(
            key=lambda tuple_kc: tuple_kc[0], reverse=True
        )  # sort based on mean

        return content, rest

    def _get_kc_details(self, kc: Dict[str, Any], history: bool) -> Tuple:
        """Extract data from a knowledge component.

        Extracts the title, mean, variance attributes from a knowledge component.
        If history is True, also extracts the timestamps from history.

        Args:
            kc:
                the knowledge component to extract the attributes from.
            history:
                 Whether to extract the timestamps from kc.

        Returns:
            A tuple containing the mean, variance, title and timestamps (if requested).

        Raises:
            TrueLearnTypeError:
                if history is True but the kc does not have a history attribute.
        """
        title = kc["title"]
        mean = kc["mean"]
        variance = kc["variance"]

        if not history:
            return mean, variance, title

        if "history" not in kc:
            raise TrueLearnTypeError(
                "Learner's knowledge does not contain history. "
                "You can use HistoryAwareKnowledgeComponents."
            )
        timestamps = [unix_to_iso(timestamp) for _, _, timestamp in kc["history"]]

        return mean, variance, title, timestamps

    @abstractmethod
    def savefig(self, file: str, **kargs):
        """Export the visualisation as a file.

        Args:
            file:
                The local file path in which to create the file.
            **kargs:
                The arguments supported by each plotter.
        """


class PlotlyBasePlotter(BasePlotter):
    """Base class for Plotly Plotters."""

    def __init__(self, title: str, xlabel: str, ylabel: str):
        """Init a Plotly Plotter.

        Args:
            title: The default title of the visualization
            xlabel: The default x label of the visualization
            ylabel: The default y label of the visualization
        """
        super().__init__()
        self.figure = go.Figure()

        # set default title, xlabel and ylabel
        self.title(title)
        self.xlabel(xlabel)
        self.ylabel(ylabel)

    def _hover_template(self, hover_fmt: Tuple, history: bool) -> str:
        """Determine what information is displayed on hover.

        Args:
            hover_fmt:
                A tuple containing some format strings that specify how to format data.
            history:
                A boolean value which determines which template to use.
                If history is True, additional information like the number of
                videos watched and the last video watched are displayed on hover.

        Returns:
            The hover template.
        """
        topic, mean, variance, number_videos, last_video = hover_fmt
        return (
            "<br>".join(
                [
                    f"Topic: {topic}",
                    f"Mean: {mean}",
                    f"Variance: {variance}",
                    f"Number of Videos Watched: {number_videos}",
                    f"Last Video Watched On: {last_video}",
                    "<extra></extra>",
                ]
            )
            if history
            else "<br>".join(
                [
                    f"Topic: {topic}",
                    f"Mean: {mean}",
                    f"Variance: {variance}",
                    "<extra></extra>",
                ]
            )
        )

    @final
    def show(self):
        """Display the visualisation in a new webpage.

        Equivalent to calling Plotly's Figure.show() method.
        """
        self.figure.show()

    @final
    def savefig(self, file: str, **kargs):
        """Export the visualisation to a file.

        Args:
            file:
                The local file path in which to create the file.
            **kargs:
                Optional supported arguments as shown below.

                This method supports saving the visualisation in various formats.
                Most platforms support the following formats:
                "png", "jpg" or "jpeg", "svg", "pdf", "html", "json".

                If you want to export a HTML file, you can optionally pass in
                    default_width:
                        The default width of the image in the HTML file.
                    default_height:
                        The default height of the image in the HTML file.

                If you want to export a JSON file, you can optionally pass in
                    pretty:
                        Whether the saved JSON representation should be pretty-printed.

                If you want to export an image file, you can optionally pass in
                    width:
                        The default width of the image.
                    height:
                        The default height of the image.

        Notes:
            You can refer to Plotly's documentation for `write_image` and `write_html`
            to find out more supported arguments.
        """
        if file.endswith(".html"):
            self.figure.write_html(file=file, **kargs)
            return
        if file.endswith(".json"):
            self.figure.write_json(file=file, **kargs)
            return

        self.figure.write_image(file=file, **kargs)

    @final
    def title(self, text: str):
        """Set the title of the figure.

        Args:
            text: The title of the figure.
        """
        self.figure.update_layout(title=text)

    @final
    def xlabel(self, text: str):
        """Set the x label of the figure.

        Args:
            text: The x label of the figure.
        """
        self.figure.update_xaxes(title_text=text)

    @final
    def ylabel(self, text: str):
        """Set the y label of the figure.

        Args:
            text: The y label of the figure.
        """
        self.figure.update_yaxes(title_text=text)


class MatplotlibBasePlotter(BasePlotter):
    """Base class for Matplotlib Plotters."""

    def __init__(self, title: str, xlabel: str, ylabel: str):
        """Init a matplotlib Plotter.

        Args:
            title: The default title of the visualization
            xlabel: The default x label of the visualization
            ylabel: The default y label of the visualization
        """
        super().__init__()
        self.fig, self.ax = plt.subplots()

        self.title(title)
        self.xlabel(xlabel)
        self.ylabel(ylabel)

    @final
    def show(self):
        """Display the figure if the caller is in an interactive environment.

        For example, if you use Jupyter Notebooks, it will be displayed.

        If you want to display the visualisations in a non-interactive environment,
        you can use `plt.show()` instead.
        """
        self.fig.show()

    @final
    def savefig(self, file: str, **kargs):
        """Export the visualisation to a file.

        Args:
            file:
                The local file path in which to create the file.
            **kargs:
                Optional supported arguments as shown below.

                This method supports saving the visualisation in various formats.
                Most platforms support the following formats:
                "png", "jpg" or "jpeg", "svg", "pdf".

                You can use `print(plotter.fig.canvas.get_supported_filetypes())`
                to find out the supported file formats in your platform.

                You can optionally pass in these arguments:
                    dpi:
                        The dpi of the image.

        Notes:
            You can refer to matplotlib's documentation for `savefig`
            to find out more supported arguments.
        """
        self.fig.savefig(fname=file, **kargs)

    @final
    def title(self, text: str):
        """Set the title of the figure.

        Args:
            text: The title of the figure.
        """
        self.ax.set_title(text)

    @final
    def xlabel(self, text: str):
        """Set the x label of the figure.

        Args:
            text: The x label of the figure.
        """
        self.ax.set_xlabel(text)

    @final
    def ylabel(self, text: str):
        """Set the y label of the figure.

        Args:
            text: The y label of the figure.
        """
        self.ax.set_ylabel(text)


def knowledge_to_dict(
    knowledge: Knowledge,
) -> Dict[Hashable, Dict[str, Union[str, float]]]:
    """Convert knowledge to a Python dictionary.

    Args:
        knowledge:
            the knowledge object to copy.

    Returns:
        A dictionary mapping of topic id to a knowledge component.
        The knowledge component is in dictionary format.
    """

    def export_as_dict_with_title(topic_id, kc):
        kc_dict = kc.export_as_dict()

        # in case that the title is None,
        # we try to use topic_id as the title
        kc_dict["title"] = kc_dict["title"] or str(topic_id)

        return kc_dict

    return {
        topic_id: export_as_dict_with_title(topic_id, kc)
        for topic_id, kc in knowledge.topic_kc_pairs()
    }


def unix_to_iso(t: float) -> str:
    """Convert a unix timestamp to an ISO-formatted date string.

    Args:
        t: value representing the POSIX timestamp.

    Returns:
        the ISO date string.
    """
    return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")


def unzip_content_dict(content_dict: List[Tuple]):
    """Unzip content dictionary.

    Args:
        content_dict:
            An non-empty list of tuples where each tuple represents a
            knowledge component.
            Inside each tuple, there are (mean, variance, title).

    Returns:
        A tuple containing (means, variances, titles).
        Where each element inside the tuple is also a tuple.
    """
    return tuple(zip(*content_dict))


def unzip_content_dict_history(content_dict: List[Tuple]):
    """Unzip content dictionary.

    Args:
        content_dict:
            An non-empty list of tuples where each tuple represents a
            knowledge component.
            Inside each tuple, there are (mean, variance, title).

    Returns:
        A tuple containing (means, variances, titles).
        Where each element inside the tuple is also a tuple.
    """
    return tuple(zip(*content_dict))
