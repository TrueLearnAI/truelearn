import random
from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    PlotlyBasePlotter,
    knowledge_to_dict,
)


class PiePlotter(PlotlyBasePlotter):
    """Provides utilities for plotting pie charts."""

    def _standardise_data(
        self,
        raw_data: Knowledge,
        history: bool = False,
        topics: Optional[Iterable[str]] = None,
    ) -> Iterable:    
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
        content: Union[Knowledge, List[Tuple[float, float, str]]],
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        title: str = "Distribution of user's skill.",
        x_label: str = "",
        y_label: str = "",
        other: bool = False,
        history: bool = False,
    ) -> Self:
        if isinstance(content, Knowledge):
            content, rest = self._standardise_data(content, history, topics)
        else:
            rest = []

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
            customdata=np.transpose(
                [titles, means, variances, number_of_videos, last_video_watched]
            ),
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

    def _get_other_data(
        self,
        rest: Iterable[Tuple[float, float, str]],
        history: bool
    ):
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
        """Maps the variance to a shade of green represented in RGB.
        
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
    """Provides utilities for plotting rose charts."""

    def plot(
        self,
        content: Union[Knowledge, List[Tuple[float, float, str]]],
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        title: str = "Distribution of user's skill.",
        x_label: str = "",
        y_label: str = "",
        other: bool = False,
        history: bool = False,
    ) -> Self:
        if isinstance(content, Knowledge):
            content, rest = self._standardise_data(content, True, topics)

        rest += content[top_n:]
        content = content[:top_n]

        random.shuffle(content)

        if other:
            content.append(self._get_other_data(rest, True))

        number_of_videos = [len(tr_data[3]) for tr_data in content]
        if other:
            number_of_videos[-1] /= len(rest)

        total_videos = sum(number_of_videos)
        widths = [(n / total_videos) * 360 for n in number_of_videos]
        thetas = [0]
        for i in range(len(widths)-1):
            thetas.append(thetas[i] + widths[i]/2 + widths[i+1]/2)
        
        variances = [tr_data[1] for tr_data in content]
        variance_min, variance_max = min(variances), max(variances)

        colours = [
            self._get_colour(v, variance_min, variance_max) for v in variances
        ]

        traces = []
        n_of_sectors = len(content)
        for i in range(n_of_sectors):
            traces.append(
                self._trace(content[i], thetas[i], widths[i], colours[i])
            )

        means = [tr_data[0] for tr_data in content]
        average_mean = sum(means) / len(means)
        traces.append(
            go.Scatterpolar(
                name="Average mean",
                r=[average_mean for _ in range(360)],
                theta=list(range(360)),
                mode='lines',
                line_color='black',
            )
        )

        self.figure = go.Figure(data=traces, layout=self._layout((title, None, None)))

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
        number_of_videos: Optional[int] = None,
    ) -> go.Barpolar:
        """Returns the Barpolar object representing a single sector.

        Args:
            tr_data:
                the data used to plot the sector. A tuple containing the mean,
                variance, title and timestamps of the topic represented by the
                sector.
            theta:
                the position of the sector alongside the angular axis,
                given in degrees.
            width:
                the width of the sector, given in degrees.
            colour:
                the colour of the sector, given as an rgb string. E.g. 'rgb(0,0,0)'.
            number_of_videos:
                the number of videos watched for the topic represented
                by the sector.
        """
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
