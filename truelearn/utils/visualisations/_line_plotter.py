from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import plotly.graph_objects as go

from ._base import unix_to_iso
from truelearn.errors import TrueLearnTypeError
from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import PlotlyBasePlotter


class LinePlotter(PlotlyBasePlotter):
    """Line Plotter.

    It can provide 2 kinds of visualization:

    - Single user + Multiple topics
    - Multiple users + Single topic

    In each mode, the x-axis represents the time
    and the y-axis represents the mean of the topic.
    """

    def __init__(
        self,
        title: str = "",
        xlabel: str = "Time",
        ylabel: str = "Mean",
    ):
        """Init a Bubble plotter.

        Args:
            title: the default title of the visualization
            xlabel: the default x label of the visualization
            ylabel: the default y label of the visualization
        """
        super().__init__(title, xlabel, ylabel)

    def _get_kc_details(self, kc, _) -> Tuple:
        """Helper function for extracting data from a knowledge component.

        Extracts the title, mean, variance for each timestamp in the knowledge
        component's history.

        Args:
            kc:
                the knowledge component to extract the attributes from.

        Returns:
            a tuple consisting of the means and variances of the topic at each
            timestamp, the title of the topic and the timestamps of when it was
            updated.

        Raises:
            TrueLearnTypeError:
                if any of the knowledge components are not history-aware.
        """
        title = kc["title"]
        means = []
        variances = []
        timestamps = []

        if "history" not in kc:
            raise TrueLearnTypeError(
                "User's knowledge does not contain history. "
                "You can use HistoryAwareKnowledgeComponents."
            )

        for mean, variance, timestamp in kc["history"]:
            means.append(mean)
            variances.append(variance)
            timestamps.append(unix_to_iso(timestamp))

        return means, variances, title, timestamps

    def plot(
        self,
        content: Union[List[Knowledge], Knowledge],
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        variance: bool = False,
    ) -> Self:
        if isinstance(content, list):
            content_dict = self._plot_multiple(content, topics)
        else:
            content_dict = self._plot_single(content, topics, top_n)

        traces = [self._trace(tr_data, variance) for tr_data in content_dict]

        self.figure.add_traces(traces)

        return self

    def _plot_single(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]],
        top_n: Optional[int],
    ):
        content_dict, _ = self._standardise_data(content, True, topics)
        return content_dict[:top_n]

    def _plot_multiple(
        self,
        content_list: Iterable[Knowledge],
        topics: Optional[Iterable[str]] = None,
    ):
        data = []
        for content in content_list:
            content_dict, _ = self._standardise_data(content, True, topics)
            # select one topic from each learner
            if content_dict:  # if user Knowledge contains at least 1 topic in topics
                data.append(content_dict[0])

        return data

    def _trace(
        self,
        tr_data: Tuple[Iterable, Iterable, str, Iterable],
        visualise_variance: bool,
    ) -> go.Scatter:
        """Returns the Scatter object representing a single line.

        Args:
            tr_data:
                the data used to plot the line. A tuple containing the y values
                and variance of each point, the name of the line (the topic or user
                it represents) and the x values of each point.
            visualise_variance:
                boolean which determines whether to make the error bars
                at each point visible or not.
        """
        means, variances, name, timestamps = tr_data

        trace = go.Scatter(
            name=name,
            x=timestamps,
            y=means,
            mode="lines+markers",
            marker={"size": 8},
            line={"width": 2},
            error_y={
                "array": variances,
                "visible": visualise_variance,
            },
        )

        return trace
