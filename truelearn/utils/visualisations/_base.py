import datetime

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Union, Tuple, List
from typing_extensions import final, Self

from plotly import (
    graph_objects as go,
    basedatatypes as bdt
)
import matplotlib.pyplot as plt

from truelearn.models import Knowledge

KnowledgeDict = Dict[str, Dict[str, Union[str, float]]]


class BasePlotter(ABC):
    """The base class of all the plotters."""

    @final
    def __init__(self):
        self.figure = None

    def _standardise_data(
        self,
        raw_data: Knowledge,
        history: bool=False,
        topics: Union[str, Iterable[str], None]=None
    ) -> List[Tuple]:
        """Converts a KnowledgeDict object to one suitable for generating visualisations.
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable used by plot.

        Args:
            raw_data:
                the learner's knowledge, represented by a Knowledge object.
            topics:
                optional iterable of topics to extract the information for from
                the knowledge object. If not specified, all topics are extracted.
            history:
                boolean which indicates whether the user has requested access to
                the knowledge component's timestamps (this allows the visualisations
                to display more information on hover such as the number of videos
                watched and the last video watched). If set to True, the Knowledge
                object must consists of HistoryAwareKnowledgeComponents, or an
                error will be raised.

        Returns:
            a data structure suitable for generating the figure via the plot() method.
        """        
        raw_data = knowledge_to_dict(raw_data)

        content = []
        for _, kc in raw_data.items():
            data = self._get_kc_details(kc, history)
            if (topics is None) or (data[2] in topics or data[2] == topics):
                content.append(data)

        content.sort(
            key=lambda data: data[0],  # sort based on mean
            reverse=True
        )

        return content

    def _get_kc_details(self, kc: Dict[str, Any], history: bool) -> Tuple:
        """Helper function for extracting data from a knowledge component.

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
            TypeError:
                if history is True but the kc does not have a history key.
        """
        title = kc['title']
        mean = kc['mean']
        variance = kc['variance']
        timestamps = []
        if history:
            try:
                for _, _, timestamp in kc['history']:
                    timestamps.append(timestamp)
                timestamps = list(map(
                    self._unix_to_iso,
                    timestamps
                ))
                data = (mean, variance, title, timestamps)
            except KeyError as err:
                raise TypeError(
                    "User's knowledge contains KnowledgeComponents. "
                    + "Expected only HistoryAwareKnowledgeComponents."
                ) from err
        else:
            data = (mean, variance, title)
        
        return data

    @final
    def _unix_to_iso(self, t: int) -> str:
        """Converts an unix timestamp to an ISO-formatted date string.

        Args:
            t: 
                int value representing the unix timestamp.
        
        Returns:
            the ISO date string.
        """
        return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")

    @abstractmethod
    def plot(
            self,
            content: Union[Knowledge, List[Tuple]],
            topics: Optional[Iterable[str]]=None,
            top_n: Optional[int]=None,
            title: str="",
            x_label: str="",
            y_label: str="",
    ) -> Self:
        """Creates a Plotly Figure object from the data.

        Args:
            content:
                the data to use to plot the visualisation. Either a Knowledge
                object (the learner's knowledge) or an iterable of tuples. Each
                tuple would represent a different topic / knowledge component.
            topics:
                the list of topics in the learner's knowledge to visualise.
                If None, all topics are visualised (unless top_n is
                specified).
            top_n:
                the number of topics to visualise. E.g. if top_n is 5, then the
                top 5 topics ranked by mean will be visualised (unless content
                is a list of tuples, in which case the first 5 tuples would be
                plotted as the list is already assumed to be in the desired order).
            title:
                the title that will be displayed on top of the visualisation.
            x_label:
                the label of the x-axis (if the visualisaition has an x-axis).
            y_label:
                the label of the y-axis (if the visualisations has a y-axis).
        """
    
    @abstractmethod
    def _static_export(self, file, format, width, height):
        """Exports the visualisation as an image file.

        Args:
            file:
                the local file path in which to create the image file.
            format:
                the format of the file. Supported formats include png,
                jpg/jpeg, webp, svg, pdf and eps (requires the poppler library to
                be installed).
            width:
                the width of the image file.
            height:
                the height of the image file.
        """

    @final
    def to_png(
            self,
            file: str,
            width: int=1000,
            height: int=600,
    ) -> None:
        """Exports the visualisation as a png file.

        Args:
            file:
                the local file path in which to create the image file (must
                end with .png).
            width:
                the width of the image file.
            height:
                the height of the image file.
        """
        self._static_export(file, "png", width, height)

    @final
    def to_jpeg(
            self,
            file: str,
            width: int=1000,
            height: int=600,
    ) -> None:
        """Exports the visualisation as a jpeg file.

        Args:
            file:
                the local file path in which to create the image file (must
                end with .jpeg).
            width:
                the width of the image file.
            height:
                the height of the image file.
        """
        self._static_export(file, "jpeg", width, height)

    @final
    def to_webp(
            self,
            file: str,
            width: int=1000,
            height: int=600,
    ) -> None:
        """Exports the visualisation as a webp file.

        Args:
            file:
                the local file path in which to create the image file (must
                end with .webp).
            width:
                the width of the image file.
            height:
                the height of the image file.
        """
        self._static_export(file, "webp", width, height)

    @final
    def to_svg(
            self,
            file: str,
            width: int=1000,
            height: int=600,
    ) -> None:
        """Exports the visualisation as an svg file.

        Args:
            file:
                the local file path in which to create the image file (must
                end with .svg).
            width:
                the width of the image file.
            height:
                the height of the image file.
        """
        self._static_export(file, "svg", width, height)

    @final
    def to_pdf(
            self,
            file: str,
            width: int=1000,
            height: int=600,
    ) -> None:
        """Exports the visualisation as a pdf file.

        Args:
            file:
                the local file path in which to create the image file (must
                end with .pdf).
            width:
                the width of the image file.
            height:
                the height of the image file.
        """
        self._static_export(file, "pdf", width, height)


class PlotlyBasePlotter(BasePlotter):
    """Provides additional methods suitable for plotting Plotly figures."""
    
    def _layout(self, layout_data: Tuple[str, str, str]) -> go.Layout:
        """Creates the Layout object for the visualisation.

        Args:
            layout_data:
                a tuple containing the title of the visualisation and the x and
                y labels.
        
        Returns:
            the Layout object created with layout_data.
        """
        title, x_label, y_label = layout_data

        layout = go.Layout(
            title=title,
            xaxis=dict(title=x_label),
            yaxis=dict(title=y_label),
        )

        return layout

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
                    "<extra></extra>"
                ]
            )
            if history else
            "<br>".join(
                [
                    f"Topic: {topic}",
                    f"Mean: {mean}",
                    f"Variance: {variance}",
                    "<extra></extra>"
                ]
            )
        )
    
    @final
    def show(self) -> None:
        """Opens the visualisation in localhost.

        Equivalent to calling Plotly's Figure.show() method.
        """
        self.figure.show()

    @final
    def _static_export(self, file: str, format: str, width, height) -> None:
        self.figure.write_image(
            file=file,
            format=format,
            width=width,
            height=height
        )

    @final
    def to_html(
        self,
        file: str,
        width: str="100%",
        height: str="100%",
    ) -> None:
        """Exports the visualisation to an HTML file.

        This will result in the visualisation being interactable.

        Args:
            file:
                the local file path in which to create the html file (must
                end with .html).
            width:
                the width of the image in the HTML file.
            height:
                the height of the image in the HTML file.
        """
        self.figure.write_html(
            file=file,
            default_width=width,
            default_height=height,
        )        


class MatplotlibBasePlotter(BasePlotter):
    """Provides additional methods suitable for plotting Matplotlib figures."""
    @final
    def show(self):
        """Opens the visualisation in a Tkinter window.

        Equivalent to calling Matplotlib.pyplot's show() method.
        """
        plt.show()

    @final
    def _static_export(self, file, format, width, height):
        plt.savefig(fname=file, format=format)


def knowledge_to_dict(
        knowledge: Knowledge,
        mapping: Optional[Dict[int, str]]=None
    ) -> KnowledgeDict:
    """Convert knowledge to a Python dictionary.
    
    Returns a copy of the knowledge object in which all the knowledge
    components have been converted to dicts.

    Args:
        knowledge:
            the knowledge object to copy.
        mapping:
            an optional dictionary intended to map the topic IDs
            in the knowledge components to a different value.
    """
    pairs = knowledge.topic_kc_pairs()

    knowledge_dict = {}
    for topic, kc in pairs:
        if mapping:
            topic = mapping[topic]
        knowledge_dict[topic] = kc.export_as_dict()

    return knowledge_dict
