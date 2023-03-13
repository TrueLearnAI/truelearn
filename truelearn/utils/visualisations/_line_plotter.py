import datetime
from typing import Dict, Iterable, Union, Tuple
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict,
)


class LinePlotter(BasePlotter):
    """Provides utilities for plotting line charts."""
    def __init__(self):
        self.figure = None
    
    def _standardise_data(
            self, raw_data: Knowledge
        ) -> Iterable[Tuple[str, Iterable, Iterable]]:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable[Tuple[str, Iterable, Iterable]] used by plot.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.

        Returns:
            A data structure usable by the plot() method to generate
            the line chart.
        """

        raw_data = knowledge_to_dict(raw_data)

        content = []
        for _, kc in raw_data.items():
            title = kc['title']
            means = []
            variances = []
            timestamps = []
            for mean, variance, timestamp in kc['history']:
                means.append(mean)
                variances.append(variance)
                timestamps.append(timestamp)
            
            timestamps = list(map(
                lambda t: datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
                timestamps
            ))
            tr_data = (title, means, variances, timestamps)
            content.append(tr_data)
        
        content.sort(
            key=lambda tr_data: tr_data[1],  # sort based on mean
            reverse=True
        )

        return content

    def plot(
            self,
            content: Iterable[Tuple[str, Iterable, Iterable]],
            top_n: int=5,
            visualise_variance: bool=True,
            title: str="Mean of user's top 5 topics over time",
            x_label: str="Time",
            y_label: str="Mean",
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
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.
        """
        if isinstance(content, Knowledge):
            content = self._standardise_data(content)
            
            # have to move this outside of the if block
            content = content[:top_n]

        traces = [self._trace(tr_data, visualise_variance) for tr_data in content]

        layout_data = self._layout((title, x_label, y_label))

        self.figure = go.Figure(
            data=traces,
            layout=layout_data
        )

        return self
    
    def _trace(
            self,
            tr_data: Tuple[str, Iterable, Iterable],
            visualise_variance
        ) -> go.Scatter:
        name, y_values, variances, x_values = tr_data

        trace = go.Scatter(
            name=name,
            x=x_values,
            y=y_values,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
            error_y=dict(
                array=variances,
                visible=visualise_variance,
            )
        )

        return trace
