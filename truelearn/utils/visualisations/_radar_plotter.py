from typing import Iterable, List, Optional
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class RadarPlotter(PlotlyBasePlotter):
    """Radar Plotter.

    Visualise the learner's knowledge by using radar plot.
    Each subject is shown in different sectors.
    """

    def __init__(
        self,
        title: str = "Mean and variance across different topics.",
        xlabel: str = "",
        ylabel: str = "",
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
        """
        content_dict, _ = self._standardise_data(content, False, topics)[:top_n]

        means = [lst[0] for lst in content_dict]
        variances = [lst[1] for lst in content_dict]
        titles = [lst[2] for lst in content_dict]

        # need to add the first element to the list again
        # otherwise, the last line will not properly shown
        means.append(means[0])
        variances.append(variances[0])
        titles.append(titles[0])

        self.figure.add_traces(
            [
                self._trace(means, titles, "Means"),
                self._trace(variances, titles, "Variances"),
            ],
        )

        self.figure.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": [
                        0,
                        int(max(max(means) + 0.001, max(variances) + 0.001) + 1),
                    ],
                }
            },
            showlegend=False,
        )

        return self

    def _trace(self, r: List[float], topics: List[str], name: str) -> go.Scatterpolar:
        """Returns a single layer in the radar chart.

        Args:
            r:
                The radial position of each point that makes up the layer.
            topics:
                The topics that need to be shown.
            name:
                The name of the trace.
        """
        return go.Scatterpolar(
            r=r,
            theta=topics,
            fill="toself",
            name=name,
            hovertemplate=self._hovertemplate("%{r}"),
        )

    def _hovertemplate(self, hoverdata: str, history: bool = False) -> str:
        """Returns the string which will be displayed when a point is hovered.

        Args:
            hoverdata:
                the format string.
            history:
                a boolean value which determines which template to use.
                Makes no difference as of yet.
        """
        return "<br>".join([f"Value: {hoverdata}", "<extra></extra>"])
