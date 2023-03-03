import datetime
from typing import Dict, Iterable, Union, Tuple
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict,
    KnowledgeDict
)


class LinePlotter(BasePlotter):
    """Provides utilities for plotting line charts."""
    def __init__(self):
        self.figure = None
    
    def clean_data(
            self, raw_data: KnowledgeDict, top_n: int
        ) -> Iterable[Tuple[str, Iterable, Iterable]]:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable[Tuple[str, Iterable, Iterable]] used by plot.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.

        Returns:
            A data structure usable by the plot() method to generate
            the line chart.
        """
        content = []
        for topic, kc in raw_data.items():
            means = []
            timestamps = []
            for mean, _, timestamp in kc['history']:
                means.append(mean)
                timestamps.append(timestamp)
            
            timestamps = list(map(
                lambda t: datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
                timestamps
            ))
            tr_data = (topic, means, timestamps)
            content.append(tr_data)
        
        content.sort(
            key=lambda tr_data: tr_data[1],  # sort based on mean
            reverse=True
        )

        return content[:top_n]        

    def plot(
            self,
            layout_data: Tuple[str, str, str],
            content: Iterable[Tuple[str, Iterable, Iterable]]
        ) -> Self:
        """Plots the line chart using the data.

        Uses content and layout_data to generate a Figure object and stores
        it into self.figure.

        Args:
            layout: a tuple of the form (title, x_label, y_label) where
              title is the what the visualisation will be named,
              x_label will be the label of the x-axis,
              y_label will be the label of the y-axis.
            content: an iterable of tuples, where each tuple is used to plot
              a line (represented through Plotly traces). Each tuple is in the
              form (name, x-values, y_values) where name is the name of the line,
              x_values are the values to plot along the x-axis and y_values
              are the values to plot along the y-axis.
        """
        traces = [self._trace(tr_data) for tr_data in content]

        layout = self._layout(layout_data)

        self.figure = go.Figure(
            data=traces,
            layout=layout
        )

        return self
    
    def _trace(self, tr_data: Tuple[str, Iterable, Iterable]) -> go.Scatter:
        name, y_values, x_values = tr_data

        trace = go.Scatter(
            name=name,
            x=x_values,
            y=y_values,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
        )

        return trace
