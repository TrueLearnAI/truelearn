import datetime
from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict
)


class RadarPlotter(BasePlotter):
    """Provides utilities for plotting radar charts."""

    def __init__(self):
        self.figure = None

    def _standardise_data(
            self, raw_data: Knowledge, history: bool
    ) -> Iterable[Tuple[Iterable, Iterable, str, Iterable]]:
        """Converts an object of KnowledgeDict type to one suitable for plot().
        
        Optional utility function that converts the dictionary representation
        of the learner's knowledge (obtainable via the knowledge_to_dict()
        function) to the Iterable[Tuple[Iterable, Iterable, str, Iterable]] 
        or Iterable[Tuple[Iterable, Iterable, str]] used by plot. Tuple type 
        depends on if user wants to visualise the history component of the
        knowledge.

        Args:
            raw_data: dictionary representation of the learner's knowledge and
              knowledge components.
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.

        Returns:
            A data structure usable by the plot() method to generate
            the radar chart.
        """
        raw_data = knowledge_to_dict(raw_data)

        content = []
        for _, kc in raw_data.items():
            title = kc['title']
            mean = kc['mean']
            variance = kc['variance']
            timestamps = []
            if history:
                try:
                    for _, _, timestamp in kc['history']:
                        timestamps.append(timestamp)
                    timestamps = list(map(
                        lambda t: datetime.datetime.utcfromtimestamp(t).strftime(
                            "%Y-%m-%d"),
                        timestamps
                    ))
                    data = (mean, variance, title, timestamps)
                except KeyError as err:
                    raise ValueError(
                        "User's knowledge contains KnowledgeComponents. "
                        + "Expected only HistoryAwareKnowledgeComponents."
                    ) from err
            else:
                data = (mean, variance, title)  # without the timestamps

            content.append(data)

        content.sort(
            key=lambda data: data[0],  # sort based on mean
            reverse=True
        )

        return content

    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            history: bool,
            top_n: int = 10,
            title: str = "Comparison of learner's top 5 subjects",
            x_label: str = "Subjects",
            y_label: str = "Mean",
    ) -> go.Scatterpolar:

        """
        Plots the radar chart using the data.

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

        layout_data = self._layout((title, x_label, y_label))

        content = content[:top_n]

        means = [lst[0] for lst in content]

        variances = [lst[1] for lst in content]

        m_min = min(min(means) - 0.001, min(variances) - 0.001)

        m_max = max(max(means) + 0.001, max(variances) + 0.001)

        titles = [lst[2] for lst in content]

        self.figure = go.Figure()

        self.figure.add_trace(go.Scatterpolar(
            r = means, 
            theta = titles, 
            fill = 'toself',
            name= 'Means',
            hovertemplate="<br>".join([
                "Mean: %{r}",
                "<extra></extra>"])

        ))

        self.figure.add_trace(go.Scatterpolar(
            r = variances, 
            theta = titles, 
            fill = 'toself',
            name= 'Variances',
            hovertemplate="<br>".join([
                "Variance: %{r}",
                "<extra></extra>"])

        ))

       
        self.figure.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, int(max(max(means) + 0.001, max(variances) + 0.001) + 1)]
            )),
        showlegend=False)
        return self

    def _trace(self):
        pass
