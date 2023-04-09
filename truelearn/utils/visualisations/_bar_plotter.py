from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class BarPlotter(PlotlyBasePlotter):
    """Bar Plotter.

    Visualise the learner's knowledge in terms of bar.
    Each subject is represented by a bar in the chart.
    The height and shade of the bar is related to the mean of the subject,
    and the error bar is related to the variance of the subject.
    """

    def __init__(
        self,
        title: str = "Comparison of learner's subjects",
        xlabel: str = "Subjects",
        ylabel: str = "Mean",
    ):
        """Init a Bar plotter.

        Args:
            title: The default title of the visualization
            xlabel: The default x label of the visualization
            ylabel: The default y label of the visualization
        """
        super().__init__(title, xlabel, ylabel)

    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        history: bool = False,
    ) -> Self:
        """Plot the graph based on the given data.

        Args:
            content:
                The Knowledge object to use to plot the visualisation.
            topics:
                The list of topics in the learner's knowledge to visualise.
                If None, all topics are visualised (unless top_n is
                specified, see below).
            top_n:
                The number of topics to visualise. E.g. if top_n is 5, then the
                top 5 topics ranked by mean will be visualised.
            history:
                Whether to utilize history information in the visualisation.
                If this is set to True, an attribute called history must be
                present in all knowledge components.
        """
        content_dict, _ = self._standardise_data(content, history, topics)[:top_n]

        means, variances, titles, *others = list(zip(*content_dict))

        if history:
            timestamps = others[0]
            number_of_videos = []
            last_video_watched = []
            for timestamp in timestamps:
                number_of_videos.append(len(timestamp))
                last_video_watched.append(timestamp[-1])
        else:
            number_of_videos = last_video_watched = [None] * len(variances)

        self.figure.add_trace(
            go.Bar(
                x=titles,
                y=means,
                width=0.5,
                marker={
                    "cmax": max(means) + 0.001,
                    "cmin": min(means) - 0.001,
                    "color": means,
                    "colorbar": {"title": "Means"},
                    "colorscale": "Greens",
                },
                error_y={
                    "type": "data",
                    "array": variances,
                    "color": "black",
                    "thickness": 4,
                    "width": 3,
                    "visible": True,
                },
                customdata=np.transpose(
                    [variances, number_of_videos, last_video_watched]  # type: ignore
                ),
                hovertemplate=self._hovertemplate(
                    (
                        "%{x}",
                        "%{y}",
                        "%{customdata[0]}",
                        "%{customdata[1]}",
                        "%{customdata[2]}",
                    ),
                    history,
                ),
            )
        )

        return self
