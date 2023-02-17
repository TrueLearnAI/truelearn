import statistics
from typing import Optional, Union
from urllib import error
from urllib import parse
from urllib import request

import orjson

Annotation = dict[str, Union[str, float, None]]
WikifierResponse = dict[str, Union[list[str], list[dict]]]


class Wikifier:
    """Used to make requests to the Wikifier API.

    Attributes:
        api_key: string representing the API key needed to make the
          request, get one from https://wikifier.org/register.html

    .. [1] Janez Brank, Gregor Leban, Marko Grobelnik. Annotating Documents
       with Relevant Wikipedia Concepts. Proceedings of the Slovenian Conference
       on Data Mining and Data Warehouses (SiKDD 2017), Ljubljana, Slovenia,
       9 October 2017.
    """

    def __init__(self, api_key: str) -> None:
        """Inits Wikifier class with api_key."""
        if isinstance(api_key, str):
            self.api_key = api_key
        else:
            raise TypeError("api_key should be a string")   

    def wikify(
            self, text: str, df_ignore: int = 50,
            words_ignore: int = 50, top_n: Optional[int] = None
        ) -> list[Annotation]:
        """Annotates input text using the Wikifier API.

        Args:
            text: string representing the text to annotate.
            df_ignore: int representing the nTopDfValuesToIgnore value from
              the Wikifier API, used to ignore frequently-occurring words.
            words_ignore: int representing the nWordsToIgnoreFromList from the
              Wikifier API, also used to ignore frequently-occurring words.
            top_n: the number of annotations to return, e.g. top_n = 5 would
              only return the top 5 annotations by pageRank.

        Returns:
            The list of annotations obtained from the Wikifier API,
            or nothing if an exception is raised.

        Raises:
            ValueError: The response from Wikifier contained an error message
              or the API key is not valid.
            RuntimeError: The HTTP request returns a status code representing
              an error
        """
        try:
            resp = self.__make_wikifier_request(
                text, df_ignore, words_ignore
            )
        except error.HTTPError as err:
            raise RuntimeError(str(err)) from err

        return self.__format_wikifier_response(resp, top_n)

    def get_cosines_mean(self, annotations: list[Annotation]) -> float:
        """Calculates the mean of the annotations' cosine values.

        Args:
            annotations: the list of Annotation objects returned by wikify()
        
        Returns:
            The mean of the cosine values in the annotations.
        """
        cosines = [ann['cosine'] for ann in annotations]
        return statistics.mean(cosines)
    
    def get_cosines_std(self, annotations: list[Annotation]) -> float:
        """Calculates the standard deviation from the annotations' cosine values.

        Args:
            annotations: the list of Annotation objects returned by wikify().
        
        Returns:
            The standard deviation of the cosine values in the annotations.
        """
        cosines = [ann['cosine'] for ann in annotations]
        return statistics.stdev(cosines)

    def __make_wikifier_request(
            self, text: str, df_ignore: int, words_ignore: int
        ) -> WikifierResponse:
        """Makes HTTP request to the Wikifier API.
        
        Requests annotations to the Wikifier API using the args from wikify(),
        loads the response JSON to a dictionary and returns all of it.

        Raises:
            ValueError: The response from Wikifier contained an error message
              or the API key is not valid.
        """
        params = {
            'text': text,
            'userKey': self.api_key,
            'nTopDfValuesToIgnore': df_ignore,
            'nWordsToIgnoreFromList': words_ignore
        }

        data = parse.urlencode(params)
        url = "https://www.wikifier.org/annotate-article?" + data
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
        """Extracts annotations from response object.

        Builds a list of annotations from the response object and simplifies
        them by getting rid of the attributes we have no interest in.

        Args:
            resp: the response from the Wikifier API as a Python dictionary.
            top_n: the number of annotations to return, e.g. top_n = 5 would
              only return the top 5 annotations by pageRank.

        Returns:
            The list of annotations obtained from the Wikifier API, sorted by
            the pageRank attribute.
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
