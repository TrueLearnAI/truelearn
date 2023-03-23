import datetime
from typing import Iterable, Tuple
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class RadarPlotter(PlotlyBasePlotter):
    """Provides utilities for plotting radar charts."""
    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            top_n: int = 10,
            title: str = "Comparison of learner's top 5 subjects",
            x_label: str = "Subjects",
            y_label: str = "Mean",
    ) -> Self:
        """
        Plots the radar chart using the data.

        Uses content and layout_data to generate a Figure object and stores
        it into self.figure.

        Args:
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.
        """
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, False)

        content = content[:top_n]

        means = [lst[0] for lst in content]

        variances = [lst[1] for lst in content]

        titles = [lst[2] for lst in content]

        self.figure = go.Figure(
            [self._trace(means, titles), self._trace(variances, titles)]
        )

        self.figure.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, int(max(max(means) + 0.001, max(variances) + 0.001) + 1)]
                )
            ),
            showlegend=False
        )

        return self

    def _trace(self, r, theta):
        return go.Scatterpolar(
            r = r, 
            theta = theta, 
            fill = 'toself',
            name= 'Variances',
            hovertemplate=self._hovertemplate("%{r}")
        )

    def _hovertemplate(self, hoverdata):
        variance = hoverdata
        return (
            "<br>".join(
                [
                    f"Variance: {variance}",
                    "<extra></extra>"
                ]
            )
        )
