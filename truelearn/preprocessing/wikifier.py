import requests
import ujson as json
import time

ERROR_KEY = u'error'

_WIKIFIER_WIKIFY_URL = u"http://www.wikifier.org/annotate-article"
# _WIKIFIER_SR_URL = "http://www.wikifier.org/get-cosine-similarity"
# _WIKIFIER_MAX_SERVER_LIMIT = 25000
# WIKIFIER_MAX_CHAR_CEILING = round(_WIKIFIER_MAX_SERVER_LIMIT * .99)  # 99% of max allowed num chars for a post request

# _ENRYCHER_URL = u"http://enrycher.ijs.si/run"

# _TAGME_WIKIFY_URL = "https://wat.d4science.org/wat/tag/tag"
# _TAGME_ENTITY_URL = u"https://wat.d4science.org/wat/title"
# _TAGME_SR_URL = u"https://wat.d4science.org/wat/relatedness/graph"


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

# used somewhere in main function
# CHUNK_SIZE = 15000
# BULK_SIZE = 50
# SILENCE_INDICATORS = ["~silence~", "~SILENCE~", "~SIL", "[SILENCE]"]
# HESITATION_INDICATORS = ["[hesitation]", "[HESITATION]"]
# UNKNOWN_INDICATORS = ["<unk>", "[UNKNOWN]", "[unknown]"]

# SPECIAL_TOKENS = set(SILENCE_INDICATORS + HESITATION_INDICATORS + UNKNOWN_INDICATORS)

DF_IGNORE_VAL = 50
WORDS_IGNORE_VAL = 50

TITLE_FIELD = u'title'
COSINE_FIELD = u'cosine'
PAGERANK_FIELD = u'pageRank'
WIKI_DATA_ID_FIELD = u'wikiDataItemId'
URL_FIELD = u'url'

STATUS_FIELD = u'status'
ANNOTATION_DATA_FIELD = u'annotation_data'

# SENTENCE_AGGREGATOR = " "  # used in partition_text
# LEN_SENTENCE_AGGR = len(SENTENCE_AGGREGATOR)

# used in the original, comes from use of pyspark
# FILEPATH_FIELD = "filepath"
# FILENAME_FIELD = "filename"
# SLUG_FIELD = "slug"
# TEXT_FIELD = "text"

# COLS = [FILEPATH_FIELD, SLUG_FIELD, TEXT_FIELD]


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


def _wikify(text, key, df_ignore=50, words_ignore=50):
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


# def wikify_data(docs, wikifier_key):
#     """ map partition function that parallaly calls the wikifier service

#     Args:
#         docs:
#         wikifier_key:

#     Yields:
#         ({key: val}): dictionary with the annotations embedded to it
#     """
#     enrichments = []
#     for part in docs:
#         annotations = _wikify(part["text"], wikifier_key, DF_IGNORE_VAL, WORDS_IGNORE_VAL)
#         part["annotations"] = annotations
#         enrichments.append(part)

#         # print("video: {}:{}:{} is completed.".format(part["slug"], part["video_id"], part["part"]))

#     return enrichments


# def get_wikifications_from_file(filepath, output_file_dir, wikifier_api_key):
#     with open(filepath) as infile:
#         lines = [json.loads(l) for l in infile.readlines() if l != ""]

#     if len(lines) == 0:
#         return {"filepath": filepath, "status": "success: blank file"}

#     annotations = list(wikify_data(lines, wikifier_api_key))

#     filename = _get_filename(filepath)

#     result_str = "\n".join([json.dumps(anno) for anno in annotations])
#     with open(output_file_dir + filename, "w") as out:
#         out.write(result_str)

#     return {"filepath": filepath, "status": "success"}
