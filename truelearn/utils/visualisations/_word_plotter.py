import warnings
import random
from typing import Iterable, Optional
from typing_extensions import Self

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import MatplotlibBasePlotter


class WordPlotter(MatplotlibBasePlotter):
    """Word cloud plotter.

    In word cloud plotter, each knowledge is represented by some words of
    a certain size and colour.

    The size of the words is proportional to the mean of the knowledge
    component the word represents.

    The color of the words is used to differentiate different knowledge
    components.
    """

    def __init__(self):
        """Init a word cloud plotter."""
        super().__init__("", "", "")

        warnings.warn(
            "This class may be removed in a future release "
            "because wordcloud library doesn't support Python 3.10+, "
            "and it is not actively maintained.",
            FutureWarning,
            stacklevel=2,
        )

    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        random_state: Optional[random.Random] = None,
    ) -> Self:
        """Plot the graph based on the given data.

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
            random_state:
                An optional random.Random object that will be used to draw word cloud.
        """
        content_dict, _ = self._standardise_data(content, False, topics)[:top_n]

        means = [lst[0] for lst in content_dict]
        titles = [lst[2].lower() for lst in content_dict]

        word_freq = {}
        for title, mean in zip(titles, means):
            word_freq[title] = int(mean * 500)

        try:
            # pylint: disable=import-outside-toplevel
            from wordcloud import WordCloud  # type: ignore
        except ImportError:
            warnings.warn(
                "Missing `wordcloud` dependency. "
                "You can install it via `pip install wordcloud`. "
                "Notice, `wordcloud` library does not support "
                "Python 3.10+ and is not actively tested against "
                "Python version >= 3.8.",
                FutureWarning,
                stacklevel=2,
            )
            return self

        wc_data = WordCloud(
            font_path="Times New Roman",
            width=800,
            height=400,
            max_words=50,
            relative_scaling=1,  # type: ignore
            normalize_plurals=False,
            background_color="white",
            random_state=random_state,
        ).generate_from_frequencies(word_freq)

        self.ax.imshow(wc_data, interpolation="bilinear")
        self.ax.axis("off")

        return self
