from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class TreePlotter(PlotlyBasePlotter):
    """Treemap plotter.

    In the treemap, each knowledge component is represented by a rectangle
    of a certain size and colour.

    The size of the rectangle is proportional to the mean of the knowledge
    component.

    The color of the rectangle is used to differentiate different knowledge
    components.
    """

    def __init__(
        self,
        title: str = "Comparison of learner's subjects",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Init a treemap plotter.

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
            go.Treemap(
                labels=titles,
                values=means,
                parents=[""] * len(titles),
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
                hovertemplate=self._hover_template(
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
