import datetime
from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict
)


class PiePlotter(BasePlotter):
    """Provides utilities for plotting bar charts."""

    def __init__(self):
        self.figure = None

    def _standardise_data(
            self, raw_data: Knowledge, history: bool
    ) -> Iterable[Tuple[Iterable, Iterable, str]]:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable[Tuple[Iterable, Iterable, str]] used by plot.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.

        Returns:
            A data structure usable by the plot() method to generate
            the bar chart.
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
                data = (mean, variance, title)  # without the timestamps

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
            other: bool = False,
            title: str = "Distribution of user's skill.",
            x_label: str = "",
            y_label: str = "",
    ) -> go.Bar:

        """
        Plots the bar chart using the data.

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
        """
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, history)

        rest = content[top_n:]
        content = content[:top_n]
        if other:
            content.append(self._get_other_data(rest, history))

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

        self.figure = go.Figure(go.Pie(
            labels=titles,
            values=means,
            customdata=np.transpose([variances, number_of_videos, last_video_watched])
            if history else
            variances,
            hovertemplate="<br>".join([
                "Topic: %{label}",
                "Mean: %{value}",
                "Variance: %{customdata[0][0]}",
                "Number of Videos Watched: %{customdata[0][1]}",
                "Last Video Watched On: %{customdata[0][2]}",
                "<extra></extra>"])
            if history else
            "<br>".join([
                "Topic: %{label}",
                "Mean: %{value}",
                "Variance: %{customdata}",
                "<extra></extra>"])
        ), layout=layout_data)

        return self

    def _get_other_data(self, rest, history):
        means = [lst[0] for lst in rest]
        variances = [lst[1] for lst in rest]
        total_mean = sum(means)
        average_variance = sum(variances) / len(rest)
        if history:
            timestamps = [lst[3] for lst in rest]
            other_data = (total_mean, average_variance, "Other", timestamps)
        else:
            other_data = (total_mean, average_variance, "Other")

        return other_data

    def _trace(self):
        pass
