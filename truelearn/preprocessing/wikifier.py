import requests
import ujson as json
import time

ERROR_KEY = u'error'

_WIKIFIER_WIKIFY_URL = u"http://www.wikifier.org/annotate-article"

DF_IGNORE_VAL = 50
WORDS_IGNORE_VAL = 50

TITLE_FIELD = u'title'
COSINE_FIELD = u'cosine'
PAGERANK_FIELD = u'pageRank'
WIKI_DATA_ID_FIELD = u'wikiDataItemId'
URL_FIELD = u'url'

STATUS_FIELD = u'status'
ANNOTATION_DATA_FIELD = u'annotation_data'


def get_wikifier_wikify_response(text, api_key, df_ignore, words_ignore):
    params = {"text": text, "userKey": api_key, "nTopDfValuesToIgnore": df_ignore,
              "nWordsToIgnoreFromList": words_ignore}
    r = requests.post(_WIKIFIER_WIKIFY_URL, params)
    if r.status_code == 200:
        resp = json.loads(r.content)
        if ERROR_KEY in resp:
            raise ValueError("error in response : {}".format(resp[ERROR_KEY]))
        return resp
    else:
        raise ValueError("http status code 200 expected, got status code {} instead".format(r.status_code))


def get_wikififier_concepts(resp, prob=0.0, top_n=None):
    annotations = list(sorted([{TITLE_FIELD: ann[TITLE_FIELD],
                                URL_FIELD: ann[URL_FIELD],
                                COSINE_FIELD: ann[COSINE_FIELD],
                                PAGERANK_FIELD: ann[PAGERANK_FIELD],
                                WIKI_DATA_ID_FIELD: ann.get(WIKI_DATA_ID_FIELD)}
                               for ann in resp.get("annotations", [])],
                              key=lambda record: record[PAGERANK_FIELD], reverse=True))

    if top_n is not None:
        annotations = list(annotations)[:top_n]

    return {
        ANNOTATION_DATA_FIELD: annotations,
        STATUS_FIELD: resp[STATUS_FIELD]
    }


def wikify(text, key, df_ignore=50, words_ignore=50):
    try:
        resp = get_wikifier_wikify_response(text, key, df_ignore, words_ignore)
        resp[STATUS_FIELD] = 'success'
    except ValueError as e:
        try:
            STATUS_ = e.message
        except:
            STATUS_ = e.args[0]
        return {
            STATUS_FIELD: STATUS_
        }
    time.sleep(0.5)
    return get_wikififier_concepts(resp)
