from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class TreePlotter(PlotlyBasePlotter):
    """Provides utilities for plotting bar charts."""
    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            history: bool,
            top_n: int = 15,
            title: str = "Comparison of learner's top 15 subjects"
    ) -> go.Bar:

        """
        Plots the bar chart using the data.

        Uses content and layout_data to generate a Figure object and stores
        it into self.figure.

        Args:
            history: a Boolean value to indicate whether or not the user wants
              to visualise the history component of the knowledge. If set to 
              True, number of videos watched by the user and the timestamp of
              the last video watched by the user will be displayed by the 
              visualisation hover text.
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.
        """
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, history)

        layout_data = self._layout((title, "", ""))

        content = content[:top_n]

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
        else:
            number_of_videos = [None for _ in variances]
            last_video_watched = [None for _ in variances]

        self.figure = go.Figure(go.Treemap(
            labels = titles,
            values = means,
            parents = ['']*len(titles),
            marker_colors = ["pink", "royalblue", "lightgray", "purple", 
                            "cyan", "lightgray", "lightblue", "lightgreen"],
            customdata=np.transpose([titles, means, variances, number_of_videos, last_video_watched]),
            hovertemplate=self._hovertemplate(
                (
                    "%{customdata[0]}",
                    "%{customdata[1]}",
                    "%{customdata[2]}",
                    "%{customdata[3]}",
                    "%{customdata[4]}"
                ),
                history
            ),
        ), layout = layout_data)

        self.figure.update_layout(margin = dict(t=50, l=25, r=25, b=25))

        return self
