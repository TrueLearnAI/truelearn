import datetime
from typing import Iterable, Union, Tuple, Optional, List
from typing_extensions import Self

import plotly.graph_objects as go
from typing_extensions import Self

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    PlotlyBasePlotter,
)


class LinePlotter(PlotlyBasePlotter):
    """Provides utilities for plotting line charts."""

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
            TypeError:
                if any of the knowledge components are not history-aware. 
        """
        title = kc['title']
        means = []
        variances = []
        timestamps = []
        try:
            for mean, variance, timestamp in kc['history']:
                means.append(mean)
                variances.append(variance)
                timestamps.append(timestamp)

            timestamps = list(map(
                self._unix_to_iso,
                timestamps
            ))

            data = (means, variances, title, timestamps)
        except KeyError as err:
            raise TypeError(
                "User's knowledge contains KnowledgeComponents. "
                + "Expected only HistoryAwareKnowledgeComponents."
            ) from err

        return data

    def plot(
        self,
        content: Iterable[Union[Knowledge, List[Tuple[Iterable, Iterable, str]]]],
        topics: Optional[Iterable[str]]=None,
        top_n: Optional[int]=None,
        variance: bool=False,
        title: str="Mean of user's top 5 topics over time",
        x_label: str="Time",
        y_label: str="Mean",
    ) -> Self:
        if isinstance(content, list):
            content = self._plot_multiple(content, topics)
        else:
            content = self._plot_single(content, topics, top_n)

        traces = [self._trace(tr_data, variance) for tr_data in content]

        layout_data = self._layout((title, x_label, y_label))

        self.figure = go.Figure(
            data=traces,
            layout=layout_data
        )

        return self

    def _plot_single(
            self,
            content: Iterable[Tuple[str, Iterable, Iterable]],
            topics: Optional[Iterable[str]],
            top_n: Optional[int]
    ):
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, True, topics)
            
        return content[:top_n]

    def _plot_multiple(
            self,
            content_list: Iterable[Union[Knowledge, Iterable[Tuple[str, Iterable, Iterable]]]],
            topics: Optional[Iterable[str]]=None,    
        ):
        data = []
        for content in content_list:
            if isinstance(content, Knowledge):
                content = self._standardise_data(content, True, topics)
                if content:  # if user Knowledge contains at least 1 topic in topics
                    data.append(content[0])
            else:
                data.append(content)

        return data

    def _trace(
        self,
        tr_data: Tuple[Iterable, Iterable, str, Iterable],
        visualise_variance: bool
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
        y_values, variances, name, x_values = tr_data

        trace = go.Scatter(
            name=name,
            x=x_values,
            y=y_values,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
            error_y=dict(
                array=variances,
                visible=visualise_variance,
            )
        )

        return trace
