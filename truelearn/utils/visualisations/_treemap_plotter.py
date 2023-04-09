from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class TreePlotter(PlotlyBasePlotter):
    """Treemap Plotter.

    Visualise the learner's knowledge in a treemap.
    Each subject is represented by a rectangle.
    The size of the rectangle is determined by the mean.
    """

    def __init__(
        self,
        title: str = "Comparison of learner's subjects",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Init a TreeMap plotter.

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
        content_dict = self._standardise_data(content, history, topics)[:top_n]

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
            go.Treemap(
                labels=titles,
                values=means,
                parents=[None] * len(titles),
                marker_colors=[
                    "pink",
                    "royalblue",
                    "lightgray",
                    "purple",
                    "cyan",
                    "lightgray",
                    "lightblue",
                    "lightgreen",
                ],
                customdata=np.transpose(
                    [
                        titles,
                        means,
                        variances,
                        number_of_videos,
                        last_video_watched,
                    ]  # type: ignore
                ),
                hovertemplate=self._hovertemplate(
                    (
                        "%{customdata[0]}",
                        "%{customdata[1]}",
                        "%{customdata[2]}",
                        "%{customdata[3]}",
                        "%{customdata[4]}",
                    ),
                    history,
                ),
            )
        )
        self.figure.update_layout(margin={"t": 50, "l": 25, "r": 25, "b": 25})

        return self
