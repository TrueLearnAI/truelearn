import requests
import ujson as json
import time


class Wikifier:
    """
    A class for making requests to the Wikifier API

    Attributes
    ----------
    api_key: string
        the API key needed to make the request, get one from
        https://wikifier.org/register.html
    
    Methods
    -------
    wikify(text, df_ignore, words_ignore)
        Annotates input text using the Wikifier API.
    """
    
    ERROR_KEY = u'error'
    _WIKIFIER_WIKIFY_URL = u"http://www.wikifier.org/annotate-article"
    TITLE_FIELD = u'title'
    COSINE_FIELD = u'cosine'
    PAGERANK_FIELD = u'pageRank'
    WIKI_DATA_ID_FIELD = u'wikiDataItemId'
    URL_FIELD = u'url'
    STATUS_FIELD = u'status'
    ANNOTATION_DATA_FIELD = u'annotation_data'

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def wikify(self, text, df_ignore=50, words_ignore=50) -> dict:
        """
        Annotates input text using the Wikifier API.

        Parameters
        ----------
        text: string
            the text to annotate
        df_ignore: int
            the nTopDfValuesToIgnore value from the Wikifier API,
            used to ignore frequently-occurring words
        words_ignore: int
            the nWordsToIgnoreFromList from the Wikifier API,
            also used to ignore frequently-occurring words.
        
        Returns
        -------
        dict
            the response from the Wikifier API as a Python dictionary
        """
        try:
            resp = self.__make_wikifier_request(
                text, df_ignore, words_ignore
            )
            resp[self.STATUS_FIELD] = 'success'
        except ValueError as e:
            try:
                self.STATUS_ = e.message
            except:
                self.STATUS_ = e.args[0]
            return {
                self.STATUS_FIELD: self.STATUS_
            }

        return self.__format_wikifier_response(resp)
    
    def __make_wikifier_request(self, text, df_ignore, words_ignore) -> dict:
        # makes request to Wikifier API, converts response to Python dictionary
        # and returns it
        params = {
            "text": text,
            "userKey": self.api_key,
            "nTopDfValuesToIgnore": df_ignore,
            "nWordsToIgnoreFromList": words_ignore
        }
        r = requests.post(self._WIKIFIER_WIKIFY_URL, params)
        if r.status_code == 200:
            resp = json.loads(r.content)
            if self.ERROR_KEY in resp:
                raise ValueError("error in response : {}".format(resp[ERROR_KEY]))
            return resp
        else:
            raise ValueError(
                "http status code 200 expected, got status code {} instead".format(r.status_code)
                )

    def __format_wikifier_response(self, resp, prob=0.0, top_n=None) -> dict:
        # simplifies the response object by only selecting
        # attributes we are interested in
        annotations = list(
            sorted(
                [{
                    self.TITLE_FIELD: ann[self.TITLE_FIELD],
                    self.URL_FIELD: ann[self.URL_FIELD],
                    self.COSINE_FIELD: ann[self.COSINE_FIELD],
                    self.PAGERANK_FIELD: ann[self.PAGERANK_FIELD],
                    self.WIKI_DATA_ID_FIELD: ann.get(self.WIKI_DATA_ID_FIELD)
                } for ann in resp.get("annotations", [])],
                key=lambda record: record[self.PAGERANK_FIELD], reverse=True
            )
        )

        if top_n is not None:
            annotations = list(annotations)[:top_n]

        return {
            self.ANNOTATION_DATA_FIELD: annotations,
            self.STATUS_FIELD: resp[self.STATUS_FIELD]
        }
