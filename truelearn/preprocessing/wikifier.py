import orjson
from urllib import request, parse, error

from typing import Optional, Union, NoReturn
Annotation = dict[str, Union[str, float, None]]
WikifierResponse = dict[str, Union[list[str], list[dict]]]


class Wikifier:
    """
    A class for making requests to the Wikifier API

    Attributes
    ----------
    api_key: str
        the API key needed to make the request, get one from
        https://wikifier.org/register.html

    Methods
    -------
    wikify(text, df_ignore, words_ignore)
        Annotates input text using the Wikifier API.

    References
    ----------
    .. [1] Janez Brank, Gregor Leban, Marko Grobelnik. Annotating Documents with Relevant Wikipedia Concepts. Proceedings of the Slovenian Conference on Data Mining and Data Warehouses (SiKDD 2017), Ljubljana, Slovenia, 9 October 2017.

    """

    def __init__(self, api_key: str) -> None:
        if isinstance(api_key, str):
            self.api_key = api_key
        else:
            raise TypeError("api_key should be a string")   

    def wikify(
            self, text: str, df_ignore: int = 50,
            words_ignore: int = 50, top_n: Optional[int] = None
        ) -> Union[list[Annotation], NoReturn]:
        """Annotates input text using the Wikifier API.

        Parameters
        ----------
        text: str
            the text to annotate.
        df_ignore: int
            the nTopDfValuesToIgnore value from the Wikifier API,
            used to ignore frequently-occurring words.
        words_ignore: int
            the nWordsToIgnoreFromList from the Wikifier API,
            also used to ignore frequently-occurring words.
        top_n: Optional[int] = None
            the number of annotations to return, e.g. top_n = 5 would only
            return the top 5 annotations by pageRank.

        Returns
        -------
        list[Annotation]
            the list of annotations obtained from the Wikifier API,
            or nothing if an exception is raised.

        Raises
        ------
        ValueError
            If the Wikifier API returns an error in the response
            or the API key is not valid.
        RuntimeError
            If the HTTP request returns a status code that represents
            an error

        """
        try:
            resp = self.__make_wikifier_request(
                text, df_ignore, words_ignore
            )
        except error.HTTPError as err:
            raise RuntimeError(str(err)) from err

        return self.__format_wikifier_response(resp, top_n)

    def __make_wikifier_request(
            self, text: str, df_ignore: int, words_ignore: int
        ) -> Union[WikifierResponse, NoReturn]:
        """Makes HTTP request to the Wikifier API, converts the JSON response
        to a Python dictionary and returns it.

        Parameters
        ----------
        text: str
            the text to annotate
        df_ignore: int
            the nTopDfValuesToIgnore value from the Wikifier API,
            used to ignore frequently-occurring words.
        words_ignore: int
            the nWordsToIgnoreFromList from the Wikifier API,
            also used to ignore frequently-occurring words.

        Returns
        -------
        Union[WikifierResponse, NoReturn]
            the response from the Wikifier API as a Python dictionary,
            or nothing if an exception is raised.

        Raises
        ------
        ValueError
            If the Wikifier API returns an error in the response
            or the API key is not valid.
        urllib.error.HTTPError
            If the HTTP request returns a status code that represents
            an error

        """
        params = {
            'text': text,
            'userKey': self.api_key,
            'nTopDfValuesToIgnore': df_ignore,
            'nWordsToIgnoreFromList': words_ignore
        }

        data = parse.urlencode(params)
        url = "http://www.wikifier.org/annotate-article?" + data
        with request.urlopen(url) as r:
            resp = r.read()
            resp = orjson.loads(resp)
            if 'error' in resp:
                raise ValueError(f"error in response : {resp['error']}")
            if 'status' in resp:
                # will trigger if key is not valid
                raise ValueError(resp['status'])
            return resp

    def __format_wikifier_response(
            self, resp: WikifierResponse, top_n: Optional[int] = None
        ) -> list[Annotation]:
        """Simplifies the response dictionary so as to include only the
        annotations, and the attributes we are interested in.

        Parameters
        ----------
        resp: WikifierResponse
            the response from the Wikifier API as a Python dictionary.
        top_n: Optional[int] = None
            the number of annotations to return, e.g. top_n = 5 would only
            return the top 5 annotations by pageRank.

        Returns
        -------
        list[Annotation]
            the list of annotations obtained from the Wikifier API

        """
        annotations = list(
            sorted(
                [{
                    'title': ann['title'],
                    'url': ann['url'],
                    'cosine': ann['cosine'],
                    'pageRank': ann['pageRank'],
                    'wikiDataItemId': ann.get('wikiDataItemId')
                } for ann in resp.get('annotations', [])],
                key=lambda record: record['pageRank'], reverse=True
            )
        )

        if top_n is not None:
            annotations = list(annotations)[:top_n]

        return annotations
