import warnings
from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import matplotlib.pyplot as plt

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import MatplotlibBasePlotter


# TODO: use a different Base for this
class WordPlotter(MatplotlibBasePlotter):
    """Provides utilities for plotting word clouds."""

    def plot(
        self,
        content: Union[Knowledge, List[Tuple[float, float, str]]],
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        *,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
    ) -> Self:
        if isinstance(content, Knowledge):
            content = self._standardise_data(content, False, topics)

        content = content[:top_n]

        means = [lst[0] for lst in content]

        titles = [lst[2].lower() for lst in content]

        word_freq = {}

        for i, t in enumerate(titles):
            word_freq[t] = int(means[i] * 500)

        try:
            # pylint: disable=import-outside-toplevel
            from wordcloud import WordCloud  # type: ignore
        except ImportError:
            warnings.warn(
                "You need to install the wordcloud library to use WordPlotter. "
                "You need to be careful with this class as it may be removed "
                "in a future release because wordcloud "
                "library doesn't support Python 3.11+.",
                FutureWarning,
                stacklevel=2,
            )
            return self

        self.figure = WordCloud(
            width=800,
            height=400,
            max_words=50,
            relative_scaling=1,  # type: ignore
            normalize_plurals=False,
            background_color="white",
        )

        self.figure.generate_from_frequencies(word_freq)

        plt.imshow(self.figure, interpolation="bilinear")
        plt.axis("off")

        return self
