import datetime
from typing import Iterable, Tuple, Union, Optional
from typing_extensions import final, Self

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

        layout_data = self._layout((title, None, None))

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
        
        if other:
            # get average number_of_videos
            number_of_videos[-1] /= len(rest)

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
        average_mean = sum(means) / len(rest)
        average_variance = sum(variances) / len(rest)
        if history:
            timestamps = []
            for lst in rest:
                timestamps += lst[3]  # concatenate timestamps

            timestamps[-1] = "N/A"  # alternatively, sort timestamps
            other_data = (average_mean, average_variance, "Other", timestamps)
        else:
            other_data = (average_mean, average_variance, "Other")

        return other_data

    def _trace():
        pass


class RosePlotter(PiePlotter):
    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            top_n: int = 5,
            other: bool = False,
            title: str = "Distribution of user's skill.",
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
            content = self._standardise_data(content, True)

        rest = content[top_n:]
        content = content[:top_n]

        if other:
            content.append(self._get_other_data(rest, True))

        number_of_videos = [len(tr_data[3]) for tr_data in content]
        if other:
            # get average number of videos watched
            number_of_videos[-1] /= len(rest)

        total_videos = sum(number_of_videos)
        widths = [(n / total_videos) * 360 for n in number_of_videos]
        thetas = [0]
        for i in range(len(widths)-1):
            thetas.append(thetas[i] + widths[i]/2 + widths[i+1]/2)

        traces = []
        for i in range(len(content)-1):
            traces.append(self._trace(content[i], thetas[i], widths[i]))
        
        traces.append(self._trace(content[-1], thetas[-1], widths[-1], number_of_videos[-1]))

        layout_data = self._layout((title, None, None))

        self.figure = go.Figure(data=traces, layout=layout_data)

        topics = [tr_data[2] for tr_data in content]

        self.figure.update_layout(
            polar = dict(
                angularaxis = dict(
                    tickmode="array",
                    tickvals=thetas,
                    ticktext=topics,
                ),
                radialaxis = dict(
                    tickangle=45,
                )
            )
        )

        return self

    def _trace(self, tr_data, theta, width, number_of_videos: Optional[int]=None):
        mean, variance, title, timestamps = tr_data
    
        if not number_of_videos:
            number_of_videos = len(timestamps)

        last_video_watched = timestamps[-1]
        
        trace = go.Barpolar(
            name=title,
            r=[mean],
            width=[width],
            hovertemplate="<br>".join([
                f"Topic: {title}",
                f"Mean: {mean}",
                f"Variance: {variance}",
                f"Number of Videos Watched: {number_of_videos}",
                f"Last Video Watched On: {last_video_watched}",
                "<extra></extra>"]),
            thetaunit='degrees',
            theta=[theta]
        )

        return trace
