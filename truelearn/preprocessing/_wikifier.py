import itertools
from typing import Optional, Union, Dict, List, Iterable, cast
from urllib import parse, request

import orjson

Annotation = Dict[str, Union[str, float, None]]
WikifierResponse = Dict[str, Union[List[Annotation], List[str]]]


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
        key_fn: str = "cosine",
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
            words_ignore:
                An int representing the nWordsToIgnoreFromList from the
                Wikifier API, also used to ignore frequently-occurring words.
            top_n:
                The number of annotations to return, e.g. top_n = 5 would
                only return the top 5 annotations sorted by keys extracted
                via key_fn. If None, return all the annotations.
            key_fn:
                A string representing the key function that is used when sorting
                the annotations. The allowed values are "cosine" and "pagerank".
                "cosine" means sorted by cosine similarity. "pagerank" means sorted
                by pagerank.

        Returns:
            The list of annotations obtained from the Wikifier API.
            An annotation is a dictionary containing five keys: "title",
            "url", "cosine", "pageRank", and "wikiDataItemId".

        Raises:
            ValueError:
                1) The response from Wikifier contained an error message.
                2) The API key is not valid.
                3) The key_fn is neither cosine nor pagerank.
            urllib.error.HTTPError:
                The HTTP request returns a status code representing
                an error.
        """
        resp = self.__make_wikifier_request(text, df_ignore, words_ignore)
        return Wikifier.__format_wikifier_response(resp, top_n, key_fn)

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
            words_ignore:
                An int representing the nWordsToIgnoreFromList from the
                Wikifier API, also used to ignore frequently-occurring words.

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

    @staticmethod
    def __format_wikifier_response(
        resp: WikifierResponse,
        top_n: Optional[int],
        key_fn: str,
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
                A string representing the key function that is used when sorting
                the annotations. The allowed values are "cosine" and "pagerank".
                "cosine" means sorted by cosine similarity. "pagerank" means sorted
                by pagerank.

        Returns:
            The list of annotations obtained from the Wikifier API, sorted by
            using the key extracted from key_fn function.
        """
        if key_fn not in ("cosine", "pagerank"):
            raise ValueError(
                "key_fn is expected to be cosine or pagerank."
                f" Got key_fn={key_fn} instead."
            )

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

        def page_rank_as_key(annotation: Annotation) -> float:
            """Return page rank from annotation.

            This method could be used to change how the annotations

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

        annotations = sorted(
            map(
                __restructure_annotation,
                cast(Iterable[Annotation], resp["annotations"]),
            ),
            key=cosine_as_key if key_fn == "cosine" else page_rank_as_key,
            reverse=True,
        )

        return list(itertools.islice(annotations, top_n))
