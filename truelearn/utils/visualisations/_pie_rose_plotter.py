from typing import Iterable, Tuple, Union, Optional
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    PlotlyBasePlotter,
    knowledge_to_dict,
)


class PiePlotter(PlotlyBasePlotter):
    """Provides utilities for plotting bar charts."""
    def _standardise_data(
        self,
        raw_data: Knowledge,
        history: bool=False,
        topics: Optional[Iterable[str]]=None,
        top_n: Optional[int]=None
    ) -> Iterable:
        """Converts a KnowledgeDict object to one suitable for generating visualisations.
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable used by plot.

        Args:
            raw_data:
                dictionary representation of the learner's knowledge and
                knowledge components.
            topics:
                optional list of topics to extract the information for from
                the knowledge object. If not specified, all topics are extracted.
            history:
                boolean which indicates whether the user wants to visualise.

        Returns:
            a data structure usable by the plot() method to generate the figure.
        """        
        raw_data = knowledge_to_dict(raw_data)

        content = []
        rest = []
        for _, kc in raw_data.items():
            data = self._get_kc_details(kc, history)
            if (topics is None) or (data[2] in topics):
                content.append(data)
            else:
                rest.append(data)

        content.sort(
            key=lambda data: data[0],  # sort based on mean
            reverse=True
        )

        return content, rest

    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            history: bool=False,
            topics: Optional[Iterable[str]]=None,
            top_n: int = 5,
            other: bool = False,
            title: str = "Distribution of user's skill.",
    ) -> Self:

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
            content, rest = self._standardise_data(content, history, topics)

        rest += content[top_n:]
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
        else:
            number_of_videos = [None for _ in variances]
            last_video_watched = [None for _ in variances]

        variance_min, variance_max = min(variances), max(variances)

        colours = [
            self._get_colour(v, variance_min, variance_max) for v in variances
        ]

        self.figure = go.Figure(go.Pie(
            labels=titles,
            values=means,
            customdata=np.transpose([titles, means, variances, number_of_videos, last_video_watched]),
            hovertemplate=self._hovertemplate(
                (
                    "%{customdata[0][0]}",
                    "%{customdata[0][1]}",
                    "%{customdata[0][2]}",
                    "%{customdata[0][3]}",
                    "%{customdata[0][4]}"
                ),
                history
            ),
            marker=dict(
                colors=colours,
            )
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

    def _get_colour(
            self, variance: float, variance_min: float, variance_max: float
        ) -> str:
        """
        Maps the variance to a shade of green represented in RGB.
        
        A darker shade represents less variance.

        Args:
            variance:
                Float value representing the variance of a knowledge component.
            variance_min:
                The smallest variance values from all the knowledge components
                we are plotting.
            variance_max:
                The largest variance values from all the knowledge components
                we are plotting.

        Returns:
            A colour string to use when plotting the figure.
        """
        variance_range = variance_max - variance_min
        variance = (variance - variance_min) / variance_range
        r = 20 + variance * 227
        g = 69 + variance * 183
        b = 38 + variance * 206

        return f"rgb({r},{g},{b})"


class RosePlotter(PiePlotter):
    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            topics: Optional[Iterable[str]]=None,
            top_n: int = 5,
            other: bool = False,
            title: str = "Distribution of user's skill.",
    ) -> Self:

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
            content, rest = self._standardise_data(content, True, topics, top_n)

        rest += content[top_n:]
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
        
        variances = [lst[1] for lst in content]
        variance_min, variance_max = min(variances), max(variances)

        colours = [
            self._get_colour(v, variance_min, variance_max) for v in variances
        ]

        traces = []
        for i in range(len(content)-1):
            traces.append(
                self._trace(content[i], thetas[i], widths[i], colours[i])
            )
        
        traces.append(
            self._trace(
                content[-1],
                thetas[-1],
                widths[-1],
                colours[-1],
                number_of_videos[-1]
            )
        )

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

    def _trace(
            self,
            tr_data,
            theta,
            width,
            colour,
            number_of_videos: Optional[int]=None,
        ):
        mean, variance, title, timestamps = tr_data
    
        if not number_of_videos:
            number_of_videos = len(timestamps)

        last_video_watched = timestamps[-1]
        
        return go.Barpolar(
            name=title,
            r=[mean],
            width=[width],
            hovertemplate=self._hovertemplate(
                (
                    title,
                    mean,
                    variance,
                    number_of_videos,
                    last_video_watched
                ),
                True
            ),
            thetaunit='degrees',
            theta=[theta],
            marker=dict(
                color=colour,
            ),
        )
