import warnings
from typing import Iterable, Optional
from typing_extensions import Self

from truelearn.models import Knowledge
from truelearn.utils.visualisations._base import (
    MatplotlibBasePlotter,
    unzip_content_dict,
)


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
            "WordPlotter may be removed in a future release "
            "because wordcloud library does not have "
            "cross-platform support for Python 3.11+, "
            "and it is not actively maintained.",
            FutureWarning,
            stacklevel=2,
        )

    def plot(
        self,
        content: Knowledge,
        topics: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        **kwargs,
    ) -> Self:
        """Plot the graph based on the given data.

        It will not draw anything if the knowledge given by the user is empty, or
        if topics and top_n make the filtered knowledge empty.

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
            **kwargs:
                Additional arguments that control the instantiation of the
                WordCloud object.

                You can pass in all the parameters supported by WordCloud object.
                You can refer to `wordcloud` documentation for all the supported
                arguments.
        """
        try:
            # pylint: disable=import-outside-toplevel
            from wordcloud import WordCloud  # type: ignore
        except ImportError:  # pragma: no cover
            # no cover because:
            #
            # In some systems, users can install wordcloud by themselves.
            # So, if we want to test this, we must specify a system
            # and python version that will never be able to install wordcloud.
            # (Therefore, all users running this test will get consistent behaviour,
            # regardless of what other packages they have installed).
            #
            # We think this significantly increases the complexity of the tests
            # as we have no control over the upstream library.
            # Thus, we decide to mark this as no cover.
            warnings.warn(
                "Missing `wordcloud` dependency. "
                "You can try installing it via `pip install wordcloud`.",
                FutureWarning,
                stacklevel=2,
            )
            return self

        content_dict, _ = self._standardise_data(content, False, topics)
        content_dict = content_dict[:top_n]

        if not content_dict:
            return self

        means, _, titles = unzip_content_dict(content_dict)
        titles = map(str.lower, titles)

        word_freq = {}
        for title, mean in zip(titles, means):
            word_freq[title] = int(mean * 500)

        # default arguments
        wc_args = {
            "width": 800,
            "height": 400,
            "max_words": 50,
            "relative_scaling": 1,
            "normalize_plurals": False,
            "background_color": "white",
            **kwargs,
        }

        wc_data = WordCloud(**wc_args).generate_from_frequencies(word_freq)

        self.ax.imshow(wc_data, interpolation="bilinear")
        self.ax.axis("off")

        return self
