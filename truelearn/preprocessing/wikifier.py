import orjson
from urllib import request, parse

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

    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

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

        """
        try:
            resp = self.__make_wikifier_request(
                text, df_ignore, words_ignore
            )
            resp['status'] = 'success'
        except ValueError as e:
            try:
                STATUS_ = e.message
            except:
                STATUS_ = e.args[0]

            raise ValueError(
                "encountered an error when making the request to Wikifier: " +
                STATUS_
            )

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
            if r.getcode() == 200:
                r = r.read()
                resp = orjson.loads(r)
                if 'error' in resp:
                    raise ValueError(f"error in response : {resp['error']}")
                if 'status' in resp:
                    raise ValueError(resp['status'])
                return resp
            else:
                raise ValueError(
                    f"http status code 200 expected, got status code {r.status_code} instead"
                )

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
