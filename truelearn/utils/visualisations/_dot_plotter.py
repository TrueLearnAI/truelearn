import datetime
import numpy as np
from typing import Dict, Iterable, Union, Tuple
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict,
    KnowledgeDict
)


class DotPlotter(BasePlotter):
    """Provides utilities for plotting dot plots."""
    def __init__(self):
        self.figure = None
    
    def _standardise_data(
            self, raw_data: Knowledge
        ) -> Iterable[Tuple[Iterable, Iterable, str, Iterable]]:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable[Tuple[str, Iterable, Iterable]] used by plot.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.

        Returns:
            A data structure usable by the plot() method to generate
            the dot plot.
        """
        raw_data = knowledge_to_dict(raw_data)

        content = []
        for _, kc in raw_data.items():
            mean=kc['mean']
            variance=kc['variance']
            title=kc['title']
            timestamps = []
            for _, _, timestamp in kc['history']:
                timestamps.append(timestamp)
            timestamps = list(map(
                lambda t: datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
                timestamps
            ))
            data = (mean, variance, title, timestamps)
            content.append(data)
        
        content.sort(
            key=lambda data: data[0],  # sort based on mean
            reverse=True
        )

        return content


    def plot(
            self,
            layout_data: Tuple[str, str, str],
            content: Iterable[Tuple[Iterable, Iterable, str]],
            top_n: int=5
        ) -> go.Scatter:

        """
        Plots the dot plot using the data.

        Uses content and layout_data to generate a Figure object and stores
        it into self.figure.

        Args:
            layout: a tuple of the form (title, x_label, y_label) where
              title is the what the visualisation will be named,
              subjects will be the label of the x-axis,
              mean will be the label of the y-axis,
              variance will be represented by the colour of the bars.
              content: an iterable of tuples, where each tuple is used to plot
              bars. Each tuple is in the form (mean, variance, url) where 
              mean is the TrueSkill rating of the user for a specific subject,
              variance represents the certainty of the model in this mean and 
              url which is used to extract the subject as a string without https
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.
        """
        if isinstance(content, Knowledge):
            content = self._standardise_data(content)
        
        content = content[:top_n]

        layout = self._layout(layout_data)

        means = [lst[0] for lst in content]

        variances = [lst[1] for lst in content]

        mean_min = min(means) - 0.001

        mean_max = max(means) + 0.001

        titles = [lst[2] for lst in content]

        timestamps = [lst[3] for lst in content]

        number_of_videos = []
        last_video_watched = []
        for timestamp in timestamps:
            number_of_videos.append(len(timestamp))
            last_video_watched.append(timestamp[-1])

        self.figure = go.Figure(data=go.Scatter(
            x=titles,
            y=means,
            marker=dict(
                size=50,
                cmax=mean_max,
                cmin=mean_min,
                color=means,
                colorbar=dict(
                    title="Means"
                ),
                colorscale="Greens" #extra colorscale - red to green
            ),
            error_y=dict(type='data',
                array=variances,
                color = 'black',
                thickness = 4,
                width = 3,
                visible=True),
            customdata=np.transpose([variances, number_of_videos, last_video_watched]),
            hovertemplate="<br>".join([
                    "Topic: %{x}",
                    "Mean: %{y}",
                    "Variance: %{customdata[0]}",
                    "Number of Videos Watched: %{customdata[1]}",
                    "Last Video Watched On: %{customdata[2]}",
                    "<extra></extra>"]),
            mode="markers"),
            layout = layout)

        return self
        
    def _trace():
        pass
