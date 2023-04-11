from typing import Iterable, List, Optional
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter, unzip_content_dict


class RadarPlotter(PlotlyBasePlotter):
    """Radar plotter.

    In the radar chart, each knowledge component is represented by two radii.

    One of the radii represents the variance of the knowledge component.

    The other one represents the mean of the knowledge component.
    """

    def __init__(
        self,
        title: str = "Mean and variance across different topics.",
    ):
        """Init a radar plotter.

        Args:
            title: The default title of the visualization
        """
        super().__init__(title, "", "")

    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
    ) -> Self:
        """Plot the graph based on the given data.

        It will not draw anything if the knowledge given by the user is empty, or
        if topics and top_n make the filtered knowledge empty.

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
        content_dict, _ = self._standardise_data(content, False, topics)
        content_dict = content_dict[:top_n]

        if not content_dict:
            return self

        means, variances, titles = unzip_content_dict(content_dict)
        means, variances, titles = list(means), list(variances), list(titles)

        # need to add the first element to the list again
        # otherwise, the last line will not properly show
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
            hovertemplate=self._hover_template("%{r}"),
        )

    def _hover_template(self, hover_fmt: str, history: bool = False) -> str:
        """Return the string that specifies the hover template.

        Args:
            hover_fmt:
                The format string.
            history:
                A boolean value which determines which template to use.
                This doesn't make any difference in radar plotter.
        """
        return "<br>".join([f"Value: {hover_fmt}", "<extra></extra>"])
