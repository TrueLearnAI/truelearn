from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import circlify
from matplotlib import (
    cm,
    colors
)
import matplotlib.pyplot as plt

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import MatplotlibBasePlotter


class BubblePlotter(MatplotlibBasePlotter):
    """Provides utilities for plotting bubble charts."""

    def plot(
        self,
        content: Union[Knowledge, List[Tuple[float, float, str]]],
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        title: str = "Comparison of learner's subjects",
        x_label: str = "",
        y_label: str = "",
    ) -> Self:
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, False, topics)

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
                ax.add_patch(
                    plt.Circle(
                        (x, y),
                        r,
                        linewidth=2,
                        color=sm.to_rgba(variances[len(variances) - 1 - i])
                    )
                )
                plt.annotate(
                    titles[len(titles) - 1 - i], 
                    (x,y) ,
                    va='center',
                    ha='center'
                )

        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_ylabel('Variance')

        return self
