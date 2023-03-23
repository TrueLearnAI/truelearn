import datetime

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Union, Tuple
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
    def _standardise_data(self, raw_data: Knowledge, history: bool=False) -> Iterable:
        """Converts a KnowledgeDict object to one suitable for generating visualisations.
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable used by plot.

        Args:
            raw_data:
                dictionary representation of the learner's knowledge and
                knowledge components.
            history:
                boolean which indicates whether the user wants to visualise

        Returns:
            a data structure usable by the plot() method to generate the figure.
        """
        raw_data = knowledge_to_dict(raw_data)

        content = []
        for _, kc in raw_data.items():
            mean = kc['mean']
            variance = kc['variance']
            title = kc['title']
            timestamps = []
            if history:
                try:
                    for _, _, timestamp in kc['history']:
                        timestamps.append(timestamp)
                    # EXTRACT THIS TO HELPER FUNCTION TO REUSE IN DIFFERENT PLACES
                    timestamps = list(map(
                        lambda t: datetime.datetime.utcfromtimestamp(t).strftime(
                            "%Y-%m-%d"),
                        timestamps
                    ))
                    data = (mean, variance, title, timestamps)
                except KeyError as err:
                    raise ValueError(
                        "User's knowledge contains KnowledgeComponents. "
                        + "Expected only HistoryAwareKnowledgeComponents."
                    ) from err
            else:
                data = (mean, variance, title)

            content.append(data)

        content.sort(
            key=lambda data: data[0],  # sort based on mean
            reverse=True
        )

        return content

    @abstractmethod
    def plot(
            self,
            content: Union[Knowledge, Iterable[Tuple]],
            title: str,
    ) -> Self:
        """Creates a Plotly Figure object from the data.

        Args:
            title:
                the name to give to the visualisation
            content:
                the data to be used to plot the visualisation.
        """


class PlotlyBasePlotter(BasePlotter):
    def __init__(self):
        self.figure = None
    
    def _layout(self, layout_data: Tuple[str, str, str]) -> go.Layout:
        """Creates the Layout object for the visualisation.

        Args:
            layout_data:
                a tuple containing the name of the visualisation
                and the x and y labels.
        
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

    def _hovertemplate(self, hoverdata, history):
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

        Equivalent to Plotly's Figure.show() method.
        """
        self.figure.show()

    @final
    def _static_export(
            self,
            path: str,
            format: str,
            width: int = 500,
            height: int = 500
    ) -> None:
        """Exports the visualisation as an image file.

        Args:
            path: the local file path in which to create the image file.
            format: the format of the file. Supported formats include png,
              jpg/jpeg, webp, svg, pdf and eps (requires the poppler library to
              be installed).
            width: the width of the image file.
            height: the height of the image file.
        """
        self.figure.write_image(
            path=path,
            format=format,
            width=width,
            height=height,
        )

    @final
    def to_png(
            self,
            path: str,
            width: int = 500,
            height: int = 500
    ) -> None:
        """Exports the visualisation as a png file.

        Args:
            path: the local file path in which to create the image file.
        """
        self._static_export(path, "png")

    @final
    def to_jpeg(
            self,
            path: str,
            width: int = 500,
            height: int = 500
    ) -> None:
        """Exports the visualisation as a jpeg file.

        Args:
            path: the local file path in which to create the image file.
        """
        self._static_export(path, "jpeg")

    @final
    def _html_export(
            self,
            path: str,
            width: int = 500,
            height: int = 500
    ) -> None:
        """Exports the visualisation to an HTML file.

        This will result in the visualisation being interactable.

        Args:
            path: the local file path in which to create the HTML file.
            width: the width of the visualisation in the HTML file.
            height: the height of the visualisation in the HTML file.
        """
        self.figure.write_html(
            path=path,
            width=width,
            height=height,
        )


class MatplotlibBasePlotter(BasePlotter):
    def __init__(self):
        self.figure = None
    
    def show(self):
        plt.show()

    #TODO: add functions for exporting matplotlib plots


def knowledge_to_dict(knowledge: Knowledge,
                      mapping: Optional[Dict[int, str]] = None) -> KnowledgeDict:
    """Convert knowledge to an easy-to-process Python dictionary.
    
    Returns a copy of the knowledge object in which all the knowledge
    components have been converted to dicts.
    Args:
        knowledge: the knowledge object to copy.
        mapping: an optional dictionary intended to map the topic IDs
          in the knowledge components to a different value.
    """
    pairs = knowledge.topic_kc_pairs()

    knowledge_dict = {}
    for topic, kc in pairs:
        if mapping:
            topic = mapping[topic]
        knowledge_dict[topic] = kc.export_as_dict()

    return knowledge_dict
