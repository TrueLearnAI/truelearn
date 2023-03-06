from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Union, Tuple
from typing_extensions import final, Self

from plotly import (
    graph_objects as go,
    basedatatypes as bdt
)

from truelearn.models import Knowledge


KnowledgeDict = Dict[str, Dict[str, Union[str, float]]]


class BasePlotter(ABC):
    """The base class of all the plotters."""
    @abstractmethod
    def _standardise_data(self, raw_data: Knowledge) -> Iterable:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable used by plot.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.

        Returns:
            a data structure usable by the plot() method to generate the figure.
        """

    @abstractmethod
    def plot(
            self,
            layout_data: Tuple[str, str, str],
            content: Union[Knowledge, Iterable[Tuple]]
        ) -> Self:
        """Creates a Plotly Figure object from the data.

        Args:
            layout_data: the labels to include in the visualisation.
            content: the data to be used to plot the visualisation.
        """

    @abstractmethod
    def _trace(self, trace_data: Tuple) -> bdt.BaseTraceType:
        """Creates a trace to incorporate in the visualisation.

        Args:
            trace_data: the data used to create the trace. This has the same
              type as the tuples in the iterable of the plot() method.
        
        Returns:
            the trace object generated from trace_data.
        """

    @final
    def _layout(self, layout_data: Tuple[str, str, str]) -> go.Layout:
        """Creates the Layout object for the visualisation.

        Args:
            layout_data: a tuple containing the name of the visualisation
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
            width: int=500,
            height: int=500
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
    def _html_export(
            self,
            path: str,
            width: int=500,
            height: int=500
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


def knowledge_to_dict(knowledge: Knowledge, mapping: Optional[Dict[int, str]] = None) -> KnowledgeDict:
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
