import requests
import ujson as json

from typing import Optional, Union, NoReturn
WikifierAnnotations = dict[str, Union[str, float, None]]


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
            self, text: str, df_ignore: int = 50, words_ignore: int = 50
        ) -> dict[str, Union[WikifierAnnotations, str]]:
        """Annotates input text using the Wikifier API.

        Parameters
        ----------
        text: str
            the text to annotate
        df_ignore: int
            the nTopDfValuesToIgnore value from the Wikifier API,
            used to ignore frequently-occurring words
        words_ignore: int
            the nWordsToIgnoreFromList from the Wikifier API,
            also used to ignore frequently-occurring words.
        
        Returns
        -------
        dict[str, Union[WikifierAnnotations, str]]
            a dictionary containing:
            1 - the list of annotations obtained from the Wikifier API
            2 - the status message of the response object

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
            return {
                'status': STATUS_
            }

        return self.__format_wikifier_response(resp)
    
    def __make_wikifier_request(
            self, text: str, df_ignore: int, words_ignore: int
        ) -> Union[WikifierAnnotations, NoReturn]:
        """Makes HTTP request to the Wikifier API, converts the JSON response to
        a Python dictionary and returns it.

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
        Union[WikifierAnnotations, NoReturn]
            the response from the Wikifier API as a Python dictionary,
            or nothing if an exception is raised

        Raises
        ------
        ValueError
            If the Wikifier API returns an error in the response.

        """
        params = {
            'text': text,
            'userKey': self.api_key,
            'nTopDfValuesToIgnore': df_ignore,
            'nWordsToIgnoreFromList': words_ignore
        }
        r = requests.post("http://www.wikifier.org/annotate-article", params)
        if r.status_code == 200:
            resp = json.loads(r.content)
            if 'error' in resp:
                raise ValueError("error in response : {}".format(resp['error']))
            return resp
        else:
            raise ValueError(
                "http status code 200 expected, got status code {} instead".format(r.status_code)
                )

    def __format_wikifier_response(
            self, resp: WikifierAnnotations, top_n: Optional[int] = None
        ) -> dict[str, Union[WikifierAnnotations, str]]:
        """Simplifies the response dictionary so as to include only the
        attributes we are interested in.

        Parameters
        ----------
        resp: WikifierAnnotations
            the response from the Wikifier API as a Python dictionary
        top_n: Optional[int] = None
            the number of annotations to return, e.g. top_n = 5 would only
            return the top 5 annotations by pageRank
        
        Returns
        -------
        dict[str, Union[WikifierAnnotations, str]]:
            a dictionary containing:
            1 - the list of annotations obtained from the Wikifier API
            2 - the status message of the response object

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

        return {
            'annotation_data': annotations,
            'status': resp['status']
        }
