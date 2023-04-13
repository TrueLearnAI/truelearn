from typing import Iterable, Optional
from typing_extensions import Self

import circlify
from matplotlib import cm, colors, patches
import matplotlib.pyplot as plt

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    MatplotlibBasePlotter,
    unzip_content_dict,
)


class BubblePlotter(MatplotlibBasePlotter):
    """Bubble plotter.

    In the bubble chart, each knowledge component is represented by a bubble
    of a certain size and shade.

    The size of the bubble is proportional to the mean of the knowledge component.

    The shade of the bubble represents the variance of the knowledge component.
    The lighter the shade, the greater the variance.
    """

    def __init__(
        self,
        title: str = "Comparison of learner's subjects",
    ):
        """Init a bubble plotter.

        Args:
            title: The default title of the visualization
        """
        super().__init__(title, "", "")

    # pylint: disable=too-many-locals
    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
    ) -> Self:
        """Plot the graph based on the given data.

        It will not draw anything if the knowledge given by the user is empty, or
        if topics and top_n make the filtered knowledge empty.

        Currently, this method requires that the mean of all the knowledge components
        of the learner Knowledge to be positive.

        Args:
            content:
                The Knowledge object to use to plot the visualisation.
            topics:
                The list of topics in the learner's knowledge to visualise.
                If None, all topics are visualised (unless top_n is
                specified, see below).
            top_n:
                The number of topics to visualise. E.g. if top_n is 5, then the
                top 5 topics ranked by mean will be visualised.
        """
        content_dict, _ = self._standardise_data(content, False, topics)
        content_dict = content_dict[:top_n]

        if not content_dict:
            return self

        means, variances, titles = unzip_content_dict(content_dict)
        circles = circlify.circlify(
            means, show_enclosure=True, target_enclosure=circlify.Circle(x=0, y=0, r=1)
        )

        self.ax.axis("off")

        # set the limit for x and y-axis
        lim = max(
            max(
                abs(circle.x) + circle.r,
                abs(circle.y) + circle.r,
            )
            for circle in circles
        )
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        cmap = cm.get_cmap("Greens_r")

        # Normalize data range to colormap range
        norm = colors.Normalize(vmin=min(variances) - 0.05, vmax=max(variances) + 0.05)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i, circle in enumerate(circles):
            if i < len(titles):
                x, y, r = circle
                self.ax.add_patch(
                    patches.Circle(
                        (x, y),
                        r,
                        linewidth=2,
                        color=sm.to_rgba(variances[len(variances) - 1 - i]),
                    )
                )
                plt.annotate(
                    titles[len(titles) - 1 - i], (x, y), va="center", ha="center"
                )

        # set up the colorbar on the right
        cbar = self.fig.colorbar(sm, ax=self.ax)
        cbar.ax.set_ylabel("Variance")

        return self
