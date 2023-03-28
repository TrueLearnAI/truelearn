import datetime
from typing import Iterable, Tuple, Optional
from typing_extensions import Self

import matplotlib.pyplot as plt

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import MatplotlibBasePlotter
from wordcloud import WordCloud


class WordPlotter(MatplotlibBasePlotter):
    """Provides utilities for plotting bar charts."""
    def plot(
            self,
            content: Iterable[Tuple[Iterable, Iterable, str]],
            topics: Optional[Iterable[str]]=None,
            top_n: int = 50
    ) -> Self:

        """
        Plots the wordcloud chart using the data.

        Uses content and layout_data to generate a Figure object and stores
        it into self.figure.

        Args:
            top_n: the number of knowledge components to visualise.
              e.g. top_n = 5 would visualise the top 5 knowledge components 
              ranked by mean.
        """
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
