# pylint: disable=missing-function-docstring,missing-class-docstring
import os
from urllib import request

import pytest

from truelearn import preprocessing

WIKIFIER_API_KEY = os.environ.get("WIKIFIER_API_KEY", None)


def test_get_values_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_mean(values) == 3.0


def test_get_values_population_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_population_std(values) == 1.4142135623730951


def test_get_values_sample_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_sample_std(values) == 1.5811388300841898


class TestWikifier:
    @pytest.mark.skipif(
        WIKIFIER_API_KEY is None,
        reason="WIKIFIER_API_KEY is missing from the environment variables. "
        "You can get one from https://wikifier.org/register.html.",
    )
    def test_wikifier(self):
        wikifier = preprocessing.Wikifier(WIKIFIER_API_KEY)  # type: ignore

        sample_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
        sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ac odio tempor orci dapibus ultrices in iaculis nunc sed.
        Adipiscing elit duis tristique sollicitudin nibh.
        At lectus urna duis convallis. Tincidunt lobortis feugiat vivamus at.
        Morbi quis commodo odio aenean sed adipiscing diam."""

        # sorted by cosine similarity
        annotation = wikifier.wikify(sample_text, top_n=1)[0]
        assert annotation["title"] == "Lorem ipsum"
        assert annotation["url"] == "http://la.wikipedia.org/wiki/Lorem_ipsum"
        assert annotation["wikiDataItemId"] == "Q152036"

        # sorted by page rank
        annotation = wikifier.wikify(sample_text, top_n=1, key_fn="pagerank")[0]
        assert annotation["title"] == "Morbus"
        assert annotation["url"] == "http://la.wikipedia.org/wiki/Morbus"
        assert annotation["wikiDataItemId"] == "Q12136"

    def test_wikifier_invalid_api_key(self):
        with pytest.raises(ValueError, match="user-key-not-found"):
            wikifier = preprocessing.Wikifier("invalid_api_key")

            sample_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
            sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
            Ac odio tempor orci dapibus ultrices in iaculis nunc sed.
            Adipiscing elit duis tristique sollicitudin nibh.
            At lectus urna duis convallis. Tincidunt lobortis feugiat vivamus at.
            Morbi quis commodo odio aenean sed adipiscing diam."""

            wikifier.wikify(sample_text)

    @pytest.mark.skipif(
        WIKIFIER_API_KEY is None,
        reason="WIKIFIER_API_KEY is missing from the environment variables. "
        "You can get one from https://wikifier.org/register.html.",
    )
    def test_wikifier_no_text(self):
        wikifier = preprocessing.Wikifier(WIKIFIER_API_KEY)  # type: ignore

        assert not wikifier.wikify("")

    @pytest.mark.skipif(
        WIKIFIER_API_KEY is None,
        reason="WIKIFIER_API_KEY is missing from the environment variables. "
        "You can get one from https://wikifier.org/register.html.",
    )
    @pytest.mark.disable_socket
    def test_wikifier_invalid_key_fn(self):
        wikifier = preprocessing.Wikifier(WIKIFIER_API_KEY)  # type: ignore

        with pytest.raises(ValueError) as excinfo:
            wikifier.wikify("Lorem ipsum", key_fn="invalid_key_fn")

        assert (
            "key_fn is expected to be cosine or pagerank. "
            "Got key_fn=invalid_key_fn instead." == str(excinfo.value)
        )

    @pytest.mark.skipif(
        WIKIFIER_API_KEY is None,
        reason="WIKIFIER_API_KEY is missing from the environment variables. "
        "You can get one from https://wikifier.org/register.html.",
    )
    @pytest.mark.disable_socket
    def test_wikifier_invalid_df_and_words_ignore(self):
        wikifier = preprocessing.Wikifier(WIKIFIER_API_KEY)  # type: ignore

        with pytest.raises(ValueError) as excinfo:
            wikifier.wikify("Lorem ipsum", df_ignore=-1)
        assert "df_ignore must >= 0. " "Got df_ignore=-1 instead." == str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            wikifier.wikify("Lorem ipsum", words_ignore=-1)
        assert "words_ignore must >= 0. " "Got words_ignore=-1 instead." == str(
            excinfo.value
        )

    @pytest.mark.disable_socket
    def test_wikifier_show_error(self, monkeypatch):
        class mock_urlopen:
            def __init__(self, *_args, **_kargs):
                ...

            def __enter__(self):
                return self

            def __exit__(self, *_args, **_kargs):
                ...

            @staticmethod
            def read():
                b = bytearray()
                b.extend(map(ord, '{"error": "this is a test"}'))
                return b

        monkeypatch.setattr(request, "urlopen", mock_urlopen)

        with pytest.raises(ValueError) as excinfo:
            preprocessing.Wikifier("You do not need API key for this test").wikify(
                "Hello World"
            )
        assert "error in response : this is a test" == str(excinfo.value)
