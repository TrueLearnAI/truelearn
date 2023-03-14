import datetime
from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict,
    KnowledgeDict
)


class BubblePlotter(BasePlotter):
    """Provides utilities for plotting bubble charts."""

    def __init__(self):
        self.figure = None

    def _standardise_data(
            self, raw_data: KnowledgeDict, history: bool
    ) -> Iterable[Tuple[Iterable, Iterable, str, Iterable]]:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable[Tuple[Iterable, Iterable, str, Iterable]] 
        or Iterable[Tuple[Iterable, Iterable, str]] used by plot. Tuple type 
        depends on if user wants to visualise the history component of the
        knowledge.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.

        Returns:
            A data structure usable by the plot() method to generate
            the bubble chart.
        """
        raw_data = knowledge_to_dict(raw_data)

        content = []
        for _, kc in raw_data.items():
            title = kc['title']
            mean = kc['mean']
            variance = kc['variance']
            timestamps = []
            if history:
                try:
                    for _, _, timestamp in kc['history']:
                        timestamps.append(timestamp)
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

    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            history: bool,
            top_n: int = 5,
            title: str = "Comparison of learner's top 5 subjects",
            x_label: str = "Mean",
            y_label: str = "Variance",
    ) -> go.Scatter:

        """
        Plots the bubble chart using the data.

        Uses content and history to generate a Figure object and stores
        it into self.figure.

        Args:
            history: a Boolean value to indicate whether or not the user wants
              to visualise the history component of the knowledge. If set to 
              True, number of videos watched by the user and the timestamp of
              the last video watched by the user will be displayed by the 
              visualisation hover text.
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.
        """
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, history)

        content = content[:top_n]

        layout_data = self._layout((title, x_label, y_label))

        means = [lst[0] for lst in content]

        variances = [lst[1] for lst in content]

        titles = [lst[2] for lst in content]

        if history:
            timestamps = [lst[3] for lst in content]
            number_of_videos = []
            last_video_watched = []
            for timestamp in timestamps:
                number_of_videos.append(len(timestamp))
                last_video_watched.append(timestamp[-1])

        self.figure = go.Figure(data=go.Scatter(
            x=means,
            y=variances,
            marker=dict(
                size=means,
                sizemode='area',
                sizeref=2. * max(means) / (200. ** 2),
                sizemin=4,
                color=variances,
                colorbar=dict(
                    title="Variance"
                ),
                colorscale="Greens",
                reversescale=True
            ),
            customdata=np.transpose([titles, number_of_videos, last_video_watched])
            if history else
            titles,
            hovertemplate="<br>".join([
                "Topic: %{customdata[0]}",
                "Mean: %{x}",
                "Variance: %{y}",
                "Number of Videos Watched: %{customdata[1]}",
                "Last Video Watched On: %{customdata[2]}",
                "<extra></extra>"])
            if history else
            "<br>".join([
                "Topic: %{customdata}",
                "Mean: %{x}",
                "Variance: %{y}",
                "<extra></extra>"]),
            mode="markers"),
            layout=layout_data)

        return self

    def _trace(self):
        pass
