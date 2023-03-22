import datetime
from typing import Iterable, Tuple

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import circlify

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    BasePlotter,
    knowledge_to_dict
)

class BubblePlotter(BasePlotter):
    """Provides utilities for plotting circle charts."""

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
            top_n: int = 15,
            title: str = "Comparison of learner's top 15 subjects",
            x_label: str = "Mean",
            y_label: str = "Variances",
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


        content = content[:top_n]
        

        means = [lst[0]*10 for lst in content]

        variances = [lst[1] for lst in content]

        titles = [lst[2] for lst in content]

        print(titles)
        print(means)
        print(variances)

        # compute circle positions:
        circles = circlify.circlify(
            means, 
            show_enclosure=True, 
            target_enclosure=circlify.Circle(x=0, y=0, r=1)
        )
        print("-------------")
        print(circles)

        # Create just a figure and only one subplot
        fig, ax = plt.subplots(figsize=(11.75,10))

        # Title
        ax.set_title(title)

        # Remove axes
        ax.axis('off')

        # Find axis boundaries
        lim = max(
            max(
                abs(circle.x) + circle.r,
                abs(circle.y) + circle.r,
            )
            for circle in circles
        )
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        # list of labels

        # Define colormap
        cmap = cm.get_cmap('Greens_r')

        # Normalize data range to colormap range
        norm = colors.Normalize(vmin=min(variances) - 0.05, vmax=max(variances) + 0.05)

        # Create ScalarMappable object
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        # print circles
        for i, circle in enumerate(circles):
            if i < len(titles):
                x, y, r = circle
                ax.add_patch(plt.Circle((x, y), r, linewidth=2,color=sm.to_rgba(variances[len(variances) - 1 - i])))
                plt.annotate(
                    titles[len(titles) - 1 - i], 
                    (x,y) ,
                    va='center',
                    ha='center'
                )

        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('Variance')
        
        plt.show()
        return self

    def _trace(self):
        pass