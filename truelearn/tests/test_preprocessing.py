# pylint: disable=missing-function-docstring
import os

import pytest

from truelearn import preprocessing

API_KEY = os.environ.get("WIKIFIER_API_KEY", None)


def test_get_values_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_mean(values) == 3.0


def test_get_values_population_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_population_std(values) == 1.4142135623730951


def test_get_values_sample_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert preprocessing.get_values_sample_std(values) == 1.5811388300841898


def test_wikifier():
    if API_KEY is None:
        return pytest.fail("WIKIFIER_API_KEY not set")

    wikifier = preprocessing.Wikifier(API_KEY)

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


def test_wikifier_invalid_api_key():
    with pytest.raises(ValueError, match="user-key-not-found"):
        wikifier = preprocessing.Wikifier("invalid_api_key")

        sample_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit,
        sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ac odio tempor orci dapibus ultrices in iaculis nunc sed.
        Adipiscing elit duis tristique sollicitudin nibh.
        At lectus urna duis convallis. Tincidunt lobortis feugiat vivamus at.
        Morbi quis commodo odio aenean sed adipiscing diam."""

        wikifier.wikify(sample_text)


def test_wikifier_no_text():
    if API_KEY is None:
        return pytest.fail("WIKIFIER_API_KEY not set")

    wikifier = preprocessing.Wikifier(API_KEY)

    assert (wikifier.wikify(""), [])


def test_wikifier_invalid_key_fn():
    if API_KEY is None:
        return pytest.fail("WIKIFIER_API_KEY not set")

    wikifier = preprocessing.Wikifier(API_KEY)

    with pytest.raises(ValueError, match="key_fn is expected to be cosine or pagerank"):
        wikifier.wikify("Lorem ipsum", key_fn="invalid_key_fn")
