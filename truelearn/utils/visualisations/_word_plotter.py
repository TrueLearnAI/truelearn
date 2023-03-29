from typing import Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import MatplotlibBasePlotter


class WordPlotter(MatplotlibBasePlotter):
    """Provides utilities for plotting word clouds."""
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

        means = [lst[0] for lst in content]

        titles = [lst[2].lower() for lst in content]

        word_freq = {}

        for i,t in enumerate(titles):
            word_freq[t] = int(means[i]*500)

        self.figure = WordCloud(
            width=800,
            height=400,
            max_words=50,
            relative_scaling=1,
            normalize_plurals=False,
            background_color="white"
        )

        self.figure.generate_from_frequencies(word_freq)

        plt.imshow(self.figure, interpolation='bilinear')
        plt.axis('off')

        return self
