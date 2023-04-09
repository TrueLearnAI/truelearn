import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Hashable
from typing_extensions import final, Self

import matplotlib.pyplot as plt
from plotly import graph_objects as go

from truelearn.errors import TrueLearnTypeError
from truelearn.models import Knowledge


KnowledgeDict = Dict[Hashable, Dict[str, Union[str, float]]]


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
                the learner's knowledge, represented by a Knowledge object.
            history:
                boolean which indicates whether the user has requested access to
                the knowledge component's timestamps (this allows the visualisations
                to display more information on hover such as the number of videos
                watched and the last video watched). If set to True, the Knowledge
                object must consists of HistoryAwareKnowledgeComponents, or an
                error will be raised.
            topics:
                optional iterable of topics to extract the information for from
                the knowledge object. If not specified, all topics are extracted.

        Returns:
            a data structure suitable for generating the figure via the plot() method.
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

        content.sort(key=lambda data: data[0], reverse=True)  # sort based on mean

        return content, rest

    def _get_kc_details(self, kc: Dict[str, Any], history: bool) -> Tuple:
        """Extract data from a knowledge component.

        Extracts the title, mean, variance attributes from a knowledge component.
        If history is True, also extracts the timestamps from its history.

        Args:
            kc:
                the knowledge component to extract the attributes from.
            history:
                boolean which determines whether to extract the timestamps.

        Returns:
            the mean, variance, title and timestamps (if requested) as a tuple.

        Raises:
            TrueLearnTypeError:
                if history is True but the kc does not have a history key.
        """
        title = kc["title"]
        mean = kc["mean"]
        variance = kc["variance"]

        if not history:
            return mean, variance, title

        if "history" not in kc:
            raise TrueLearnTypeError(
                "User's knowledge does not contain history. "
                "You can use HistoryAwareKnowledgeComponents."
            )
        timestamps = [unix_to_iso(timestamp) for _, _, timestamp in kc["history"]]

        return mean, variance, title, timestamps

    @abstractmethod
    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
    ) -> Self:
        """Creates a visualisation object from the data.

        Args:
            content:
                The Knowledge object to use to plot the visualisation.
            topics:
                The list of topics in the learner's knowledge to visualise.
                If None, all topics are visualised (unless top_n is
                specified).
            top_n:
                The number of topics to visualise. E.g. if top_n is 5, then the
                top 5 topics ranked by mean will be visualised if content is a
                Knowledge object or just the first 5 topics if content is a list
                (in which case content is assumed to be already sorted).
        """

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
            title: the default title of the visualization
            xlabel: the default x label of the visualization
            ylabel: the default y label of the visualization
        """
        super().__init__()
        self.figure = go.Figure()

        # set default title, xlabel and ylabel
        self.title(title)
        self.xlabel(xlabel)
        self.ylabel(ylabel)

    # TODO: remove this from Base in the next version
    def _hovertemplate(self, hoverdata: Tuple, history: bool) -> str:
        """Determines what information is displayed on hover in a dynamic setting.

        Args:
            hoverdata:
                a tuple containing the data to display on hover.
            history:
                a boolean value which determines which template to use.
                If history is True, additional information like the number of
                videos watched and the last video watched are displayed on hover.

        Returns:
            the string to display when the specific trace is hovered.
        """
        topic, mean, variance, number_videos, last_video = hoverdata
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
    def show(self) -> None:
        """Display the visualisation in a new web page.

        Equivalent to calling Plotly's Figure.show() method.
        """
        self.figure.show()

    @final
    def savefig(self, file: str, **kargs) -> None:
        """Export the visualisation to a file.

        Args:
            file:
                The local file path in which to create the file.
            **kargs:
                Optional supported arguments as shown below.

                This method supports saving the visualisation in various formats.
                Most platforms support the following formats:
                "png", "jpg" or "jpeg", "svg", "pdf", "html".

                If you want to export a HTML file, you can optionally pass in
                    default_width:
                        the default width of the image in the HTML file.
                    default_height:
                        the default height of the image in the HTML file.

                If you want to export an image file, you can optionally pass in
                    width:
                        the default width of the image.
                    height:
                        the default height of the image.

        Notes:
            You can refer to Plotly's documentation for `write_image` and `write_html`
            to find out more supported arguments for image and html files.
        """
        if file.endswith(".html"):
            self.figure.write_html(file=file, **kargs)
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
            title: the default title of the visualization
            xlabel: the default x label of the visualization
            ylabel: the default y label of the visualization
        """
        super().__init__()
        self.fig, self.ax = plt.subplots()

        self.title(title)
        self.xlabel(xlabel)
        self.ylabel(ylabel)

    @final
    def show(self):
        """Display the figure if in interactive mode."""
        self.fig.show()

    @final
    def set_size_inches(self, width: float, height: float):
        """Set the figure size in inches.

        Args:
            width: width
            height: height
        """
        self.fig.set_size_inches(width, height)

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
            to find out more supported arguments for saving image files.
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


def knowledge_to_dict(knowledge: Knowledge) -> KnowledgeDict:
    """Convert knowledge to a Python dictionary.

    Returns a copy of the knowledge object in which all the knowledge
    components have been converted to dicts.

    Args:
        knowledge:
            the knowledge object to copy.
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


def unix_to_iso(t: int) -> str:
    """Convert an unix timestamp to an ISO-formatted date string.

    Args:
        t: int value representing the unix timestamp.

    Returns:
        the ISO date string.
    """
    return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
