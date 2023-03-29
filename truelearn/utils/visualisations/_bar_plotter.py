from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class BarPlotter(PlotlyBasePlotter):
    """Provides utilities for plotting bar charts."""

    def plot(
        self,
        content: Union[Knowledge, List[Tuple[float, float, str]]],
        history: bool=False,
        topics: Optional[Iterable[str]]=None,
        top_n: Optional[int]=None,
        title: str="Comparison of learner's top subjects",
        x_label: str="Subjects",
        y_label: str="Mean",
    ) -> Self:
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, history, topics)

        layout_data = self._layout((title, x_label, y_label))

        content = content[:top_n]

        means = [lst[0] for lst in content]

        variances = [lst[1] for lst in content]

        mean_min = min(means) - 0.001

        mean_max = max(means) + 0.001

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

        self.figure = go.Figure(go.Bar(
            x=titles,
            y=means,
            width=0.5,
            marker=dict(
                cmax=mean_max,
                cmin=mean_min,
                color=means,
                colorbar=dict(
                    title="Means"
                ),
                colorscale="Greens"
            ),
            error_y=dict(type='data',
                         array=variances,
                         color='black',
                         thickness=4,
                         width=3,
                         visible=True),
            customdata=np.transpose([variances, number_of_videos, last_video_watched]),
            hovertemplate=self._hovertemplate(
                (
                    "%{x}",
                    "%{y}",
                    "%{customdata[0]}",
                    "%{customdata[1]}",
                    "%{customdata[2]}"
                ),
                history
            )
        ), layout=layout_data)

        return self
