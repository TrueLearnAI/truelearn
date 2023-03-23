import datetime
from typing import Iterable, Tuple

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import circlify

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import MatplotlibBasePlotter

class BubblePlotter(MatplotlibBasePlotter):
    """Provides utilities for plotting circle charts."""
    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
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
            content = self._standardise_data(content, False)
        content = content[:top_n]

        means = [lst[0]*10 for lst in content]

        variances = [lst[1] for lst in content]

        titles = [lst[2] for lst in content]

        circles = circlify.circlify(
            means, 
            show_enclosure=True, 
            target_enclosure=circlify.Circle(x=0, y=0, r=1)
        )

        fig, ax = plt.subplots(figsize=(11.75,10))

        ax.set_title(title)

        ax.axis('off')

        lim = max(
            max(
                abs(circle.x) + circle.r,
                abs(circle.y) + circle.r,
            )
            for circle in circles
        )
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        cmap = cm.get_cmap('Greens_r')

        # Normalize data range to colormap range
        norm = colors.Normalize(vmin=min(variances) - 0.05, vmax=max(variances) + 0.05)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

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

        return self
