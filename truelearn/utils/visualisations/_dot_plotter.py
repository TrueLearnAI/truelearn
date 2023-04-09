from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class DotPlotter(PlotlyBasePlotter):
    """Dot Plotter.

    Visualise the learner's knowledge in terms of dot.
    Each subject is represented by a dot in the chart.
    The height and shade of the dot is related to mean of the subject,
    and the error bar is related to the variance of the subject.
    """

    def __init__(
        self,
        title: str = "Comparison of learner's subjects",
        xlabel: str = "Subjects",
        ylabel: str = "Mean",
    ):
        """Init a Dot plotter.

        Args:
            title: the default title of the visualization
            xlabel: the default x label of the visualization
            ylabel: the default y label of the visualization
        """
        super().__init__(title, xlabel, ylabel)

    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        history: bool = False,
    ) -> Self:
        content_dict, _ = self._standardise_data(content, history, topics)[:top_n]

        means, variances, titles, *others = list(zip(*content_dict))

        # TODO: potential duplication
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
            go.Scatter(
                x=titles,
                y=means,
                marker={
                    "size": 50,
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
                mode="markers",
            )
        )

        return self
