import random
from typing import Iterable, Optional, Tuple, List
from typing_extensions import Self

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class PiePlotter(PlotlyBasePlotter):
    """Pie Plotter."""

    def __init__(
        self,
        title: str = "Distribution of user's skill.",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Init a Pie plotter.

        Args:
            title: the default title of the visualization
            xlabel: the default x label of the visualization
            ylabel: the default y label of the visualization
        """
        super().__init__(title, xlabel, ylabel)

    # pylint: disable=too-many-locals,too-many-arguments
    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        other: bool = False,
        history: bool = False,
    ) -> Self:
        content_dict, rest = self._standardise_data(content, history, topics)
        rest += content_dict[top_n:]
        content_dict = content_dict[:top_n]

        if other:
            content_dict.append(self._summarize_other(rest, history))

        means, variances, titles, *others = list(zip(*content_dict))

        if history:
            timestamps = others[0]
            number_of_videos = []
            last_video_watched = []
            for timestamp in timestamps:
                number_of_videos.append(len(timestamp))
                last_video_watched.append(timestamp[-1])

            if other:
                # get average number_of_videos for others
                number_of_videos[-1] /= len(rest)
        else:
            number_of_videos = last_video_watched = [None] * len(variances)

        variance_min, variance_max = min(variances), max(variances)

        colours = [self._get_colour(v, variance_min, variance_max) for v in variances]

        self.figure.add_trace(
            go.Pie(
                labels=titles,
                values=means,
                customdata=np.transpose(
                    np.array(
                        [
                            titles,
                            means,
                            variances,
                            number_of_videos,
                            last_video_watched,
                        ],
                        dtype=object,
                    )
                ),
                hovertemplate=self._hovertemplate(
                    (
                        "%{customdata[0][0]}",
                        "%{customdata[0][1]}",
                        "%{customdata[0][2]}",
                        "%{customdata[0][3]}",
                        "%{customdata[0][4]}",
                    ),
                    history,
                ),
                marker={
                    "colors": colours,
                },
            )
        )

        return self

    def _summarize_other(self, rest: List[Tuple], history: bool):
        means = [lst[0] for lst in rest]
        variances = [lst[1] for lst in rest]

        average_mean = sum(means) / len(rest)
        average_variance = sum(variances) / len(rest)

        if history:
            timestamps = [lst[3] for lst in rest]
            # timestamps[-1] = "N/A"  # alternatively, sort timestamps
            other_data = (average_mean, average_variance, "Other", timestamps)
        else:
            other_data = (average_mean, average_variance, "Other")

        return other_data

    def _get_colour(
        self, variance: float, variance_min: float, variance_max: float
    ) -> str:
        """Maps the variance to a shade of green represented in RGB.

        A darker shade represents less variance.

        Args:
            variance:
                Float value representing the variance of a knowledge component.
            variance_min:
                The smallest variance values from all the knowledge components
                we are plotting.
            variance_max:
                The largest variance values from all the knowledge components
                we are plotting.

        Returns:
            A colour string to use when plotting the figure.
        """
        variance_range = variance_max - variance_min
        variance = (variance - variance_min) / variance_range
        r = 20 + variance * 227
        g = 69 + variance * 183
        b = 38 + variance * 206

        return f"rgb({r},{g},{b})"


class RosePlotter(PiePlotter):
    """Rose Pie Plotter."""

    # TODO: respect history
    # pylint: disable=too-many-locals,too-many-arguments
    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        other: bool = False,
        history: bool = True,
    ) -> Self:
        content_dict, rest = self._standardise_data(content, True, topics)
        rest += content_dict[top_n:]
        content_dict = content_dict[:top_n]

        random.shuffle(content_dict)

        if other:
            content_dict.append(self._summarize_other(rest, True))

        number_of_videos = [len(tr_data[3]) for tr_data in content_dict]
        if other:
            number_of_videos[-1] /= len(rest)  # type: ignore

        total_videos = sum(number_of_videos)

        widths = [(n / total_videos) * 360 for n in number_of_videos]
        thetas = [0.0]
        for i in range(len(widths) - 1):
            thetas.append(thetas[i] + widths[i] / 2 + widths[i + 1] / 2)

        variances = [tr_data[1] for tr_data in content_dict]
        variance_min, variance_max = min(variances), max(variances)

        colours = [self._get_colour(v, variance_min, variance_max) for v in variances]

        traces = []
        n_of_sectors = len(content_dict)
        for i in range(n_of_sectors):
            traces.append(
                self._trace(content_dict[i], thetas[i], widths[i], colours[i])
            )

        means = [tr_data[0] for tr_data in content_dict]
        average_mean = sum(means) / len(means)
        traces.append(
            go.Scatterpolar(
                name="Average mean",
                r=[average_mean for _ in range(360)],
                theta=list(range(360)),
                mode="lines",
                line_color="black",
            )
        )

        self.figure.add_traces(data=traces)

        topics = [tr_data[2] for tr_data in content_dict]
        self.figure.update_layout(
            polar={
                "angularaxis": {
                    "tickmode": "array",
                    "tickvals": thetas,
                    "ticktext": topics,
                },
                "radialaxis": {
                    "tickangle": 45,
                },
            }
        )

        return self

    def _trace(
        self,
        tr_data: Tuple[float, float, str, List[Tuple[float, float, float]]],
        theta: float,
        width: float,
        colour: str,
    ) -> go.Barpolar:
        """Returns the Barpolar object representing a single sector.

        Args:
            tr_data:
                the data used to plot the sector. A list of tuples containing the mean,
                variance, title and timestamps of the topic represented by the sector.
            theta:
                the position of the sector alongside the angular axis,
                given in degrees.
            width:
                the width of the sector, given in degrees.
            colour:
                the colour of the sector, given as an rgb string. E.g. 'rgb(0,0,0)'.
        """
        mean, variance, title, timestamps = tr_data
        number_of_videos = len(timestamps)
        last_video_watched = timestamps[-1]

        return go.Barpolar(
            name=title,
            r=[mean],
            width=[width],
            hovertemplate=self._hovertemplate(
                (title, mean, variance, number_of_videos, last_video_watched), True
            ),
            thetaunit="degrees",
            theta=[theta],
            marker={
                "color": colour,
            },
        )
