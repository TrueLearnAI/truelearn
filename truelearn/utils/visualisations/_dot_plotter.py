from typing import Dict, Iterable, Union, Tuple
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict,
    KnowledgeDict
)


class DotPlotter(BasePlotter):
    """Provides utilities for plotting dot plots."""
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
            the dot plot.
        """
        content = []
        for _, kc in raw_data.items():
            title = kc['title']
            mean = kc['mean']
            variance = kc['variance']
            data = (title, mean, variance)
            content.append(data)
        
        content.sort(
            key=lambda data: data[1],  # sort based on mean
            reverse=True
        )
        print(content[:top_n])

        return content[:top_n]


    def plot(
            self,
            layout_data: Tuple[str, str, str],
            content: Iterable[Tuple[Iterable, Iterable, str]]
        ) -> go.Scatter:

        """
        Plots the dot plot using the data.

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

        layout = self._layout(layout_data)

        means = [lst[1] for lst in content]

        variances = [lst[2] for lst in content]

        var_min = min(variances) - 0.001

        var_max = max(variances) + 0.001

        titles = [lst[0] for lst in content]

        self.figure = go.Figure(data=go.Scatter(
            x=titles,
            y=means,
            marker=dict(
                size=25,
                cmax=var_max,
                cmin=var_min,
                color=variances,
                colorbar=dict(
                    title="Variance"
                ),
                colorscale="Viridis"
            ),
            customdata=variances,
            hovertemplate="<br>".join([
                    "Topic: %{x}",
                    "Mean: %{y}",
                    "Variance: %{customdata}",
                    "<extra></extra>"]),
            mode="markers"),
            layout = layout)

        return self
