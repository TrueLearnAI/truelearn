from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class RadarPlotter(PlotlyBasePlotter):
    """Provides utilities for plotting radar charts."""

    def plot(
        self,
        content: Union[Knowledge, List[Tuple[float, float, str]]],
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        *,
        title: str = "Mean and variance across different topics.",
        x_label: str = "",
        y_label: str = "",
    ) -> Self:
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, False, topics)

        content = content[:top_n]

        means = [lst[0] for lst in content]

        variances = [lst[1] for lst in content]

        titles = [lst[2] for lst in content]

        self.figure = go.Figure(
            [self._trace(means, titles), self._trace(variances, titles)],
            layout=self._layout((title, None, None)),
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

    def _trace(self, r: List[float], theta: List[float]) -> go.Scatterpolar:
        """Returns a single layer in the radar chart.

        Args:
            r:
                the radial position of each point that makes up the layer.
            theta:
                the angular position of each point that makes up the layer.
        """
        r.append(r[0])
        theta.append(theta[0])
        return go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name="Variances",
            hovertemplate=self._hovertemplate("%{r}"),
        )

    def _hovertemplate(self, hoverdata: float, history: bool = False) -> str:
        """Returns the string which will be displayed when a point is hovered.

        Args:
            hoverdata:
                the variance value to embed in the string.
            history:
                a boolean value which determines which template to use.
                Makes no difference as of yet.
        """
        variance = hoverdata
        return "<br>".join([f"Variance: {variance}", "<extra></extra>"])
