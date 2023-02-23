from typing import Optional, Union, Dict, List, Iterable, Callable, cast
from urllib import parse, request

import orjson

Annotation = Dict[str, Union[str, float, None]]
WikifierResponse = Dict[str, Union[List[Annotation], List[str]]]


def page_rank_as_key(annotation: Annotation) -> float:
    """Return page rank from annotation.

    Args:
        annotation: An annotation from the wikifier API.

    Returns:
        A float indicating the page rank of the item.
    """
    return cast(float, annotation["pageRank"])


def cosine_as_key(annotation: Annotation) -> float:
    """Return cosine from annotation.

    Args:
        annotation: An annotation from the wikifier API.

    Returns:
        A float indicating the cosine of the item.
    """
    return cast(float, annotation["cosine"])


class Wikifier:
    """A client that makes requests to the wikifier API."""

    def __init__(self, api_key: str) -> None:
        """Init Wikifier class with api_key.

        Args:
            api_key:
                A string representing the API key needed to make the
                request. Get one from https://wikifier.org/register.html.
        """
        self.__api_key = api_key

    def wikify(
        self,
        text: str,
        *,
        df_ignore: int = 50,
        words_ignore: int = 50,
        top_n: Optional[int] = None,
        key_fn: Callable[[Annotation], Union[float, str]] = cosine_as_key,
    ) -> List[Annotation]:
        """Annotate input text using the Wikifier API.

        Args:
            text:
                A string representing the text to annotate.
            *:
                Use to reject other positional arguments.
            df_ignore:
                An int representing the nTopDfValuesToIgnore value from
                the Wikifier API, used to ignore frequently-occurring words.
                Defaults to 50.
            words_ignore:
                An int representing the nWordsToIgnoreFromList from the
                Wikifier API, also used to ignore frequently-occurring words.
                Defaults to 50.
            top_n:
                The number of annotations to return, e.g. top_n = 5 would
                only return the top 5 annotations sorted by keys extracted
                via key_fn. If None, return all the annotations.
                Defaults to None.
            key_fn:
                A function that takes in an annotation and returns a value that
                can be used to sort different annotations.
                Defaults to cosine_as_key, which sorts the annotations based on
                their cosine values.

        Returns:
            The list of annotations obtained from the Wikifier API.
            An annotation is a dictionary containing five keys: "title",
            "url", "cosine", "pageRank", and "wikiDataItemId".

        Raises:
            ValueError:
                The response from Wikifier contained an error message
                or the API key is not valid.
            urllib.error.HTTPError:
                The HTTP request returns a status code representing
                an error.
        """
        resp = self.__make_wikifier_request(text, df_ignore, words_ignore)
        return self.__format_wikifier_response(resp, top_n, key_fn)

    def __make_wikifier_request(
        self, text: str, df_ignore: int, words_ignore: int
    ) -> WikifierResponse:
        """Make HTTP request to the Wikifier API.

        Request annotations to the Wikifier API using the args from wikify(),
        load the response JSON to a dictionary and return all of it.

        Args:
            text:
                A string representing the text to annotate.
            df_ignore:
                An int representing the nTopDfValuesToIgnore value from
                the Wikifier API, used to ignore frequently-occurring words.
                Defaults to 50.
            words_ignore:
                An int representing the nWordsToIgnoreFromList from the
                Wikifier API, also used to ignore frequently-occurring words.
                Defaults to 50.

        Raises:
            ValueError: The response from Wikifier contained an error message
              or the API key is not valid.

        Returns:
            The http response.
        """
        params = {
            "text": text,
            "userKey": self.__api_key,
            "nTopDfValuesToIgnore": df_ignore,
            "nWordsToIgnoreFromList": words_ignore,
        }

        data = parse.urlencode(params, encoding="utf-8")
        url = f"https://www.wikifier.org/annotate-article?{data}"

        # nosec because we know `url` is a valid https url
        with request.urlopen(url) as r:  # nosec
            resp = orjson.loads(r.read().decode("utf-8"))
            if "error" in resp:
                raise ValueError(f"error in response : {resp['error']}")
            if "status" in resp:
                # will trigger if key is not valid
                raise ValueError(resp["status"])
            return resp

    def __format_wikifier_response(
        self,
        resp: WikifierResponse,
        top_n: Optional[int],
        key_fn: Callable[[Annotation], Union[float, str]],
    ) -> List[Annotation]:
        """Extract annotations from response object.

        Build a list of annotations from the response object and simplify
        them by getting rid of the attributes we have no interest in.

        Args:
            resp:
                The response from the Wikifier API as a Python dictionary.
            top_n:
                The number of annotations to return, e.g. top_n = 5 would
                only return the top 5 annotations sorted by keys extracted
                via key_fn.
            key_fn:
                A function that takes in an annotation and returns a value that
                can be used to sort different annotations.

        Returns:
            The list of annotations obtained from the Wikifier API, sorted by
            using the key extracted from key_fn function.
        """

        def __restructure_annotation(
            annotation: Annotation,
        ) -> Annotation:
            return {
                "title": annotation["title"],
                "url": annotation["url"],
                "cosine": annotation["cosine"],
                "pageRank": annotation["pageRank"],
                "wikiDataItemId": annotation["wikiDataItemId"],
            }

        annotations = list(
            sorted(
                map(
                    __restructure_annotation,
                    cast(Iterable[Annotation], resp["annotations"]),
                ),
                key=key_fn,
                reverse=True,
            )
        )

        if top_n is not None:
            return annotations[:top_n]

        return annotations
