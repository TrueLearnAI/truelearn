from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine

from analyses.truelearn_experiments.utils import get_summary_stats, get_topic_dict, decode_vector


def get_tfidf(tf, df, N):
    return np.log(1 + tf) * np.log(N / (1 + df))


def _get_doc_sim(dict_p, dict_t, is_user_profile=False, df_mapping=None):
    topics = set(dict_p.keys()) | set(dict_t.keys())
    vec_p, vec_t = np.zeros(len(topics)), np.zeros(len(topics))

    for idx, topic in enumerate(topics):
        tp = dict_p.get(topic, 0.)
        tt = dict_t.get(topic, 0.)

        # create TFIDF tranformation
        if df_mapping:
            tp = get_tfidf(tp, df_mapping.value.get(str(int(topic)), 0), df_mapping.value["N"])
            tt = get_tfidf(tt, df_mapping.value.get(str(int(topic)), 0), df_mapping.value["N"])

        vec_p[idx] = tp
        vec_t[idx] = tt

    # check if both vectors are non origin
    if sum(vec_p) > 0. and sum(vec_t) > 0.:
        if is_user_profile and not df_mapping:
            vec_p = vec_p / sum(vec_p)

        cos_sim = 1. - cosine(vec_p, vec_t)
    # if one is origin:
    else:
        cos_sim = 0.

    return cos_sim


def update_topic_dict(retained_dict, current_dict, is_user):
    if not is_user:
        return current_dict
    else:
        for k, v in current_dict.items():
            retained_dict[k] += float(v)

    return retained_dict


def predict_and_model_cbf(records, is_user_profile=False, df_mapping=None, start_event=0):
    actual = []  # as the draw probability cant be zero
    predicted = []
    topic_dict_p = defaultdict(float)

    for idx, event in enumerate(records):
        _, _, _, _, _, _, topic_vec_t, label_t = decode_vector(event)
        topic_dict_t = get_topic_dict(topic_vec_t)

        # if idx == 0:
        #     topic_dict_p = update_topic_dict(topic_dict_p, topic_dict_t, is_user_profile)
        #     continue

        sim = _get_doc_sim(topic_dict_p, topic_dict_t, is_user_profile, df_mapping)

        topic_dict_p = update_topic_dict(topic_dict_p, topic_dict_t, is_user_profile)

        if idx >= start_event:
            actual.append(int(label_t))
            predicted.append(float(sim))

    return actual, predicted


def content_based_filtering_model(records, threshold=.5, start_event=0):
    num_records = float(len(records))

    # for every event
    actual, pred_probs = predict_and_model_cbf(records, start_event=start_event)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, _ = get_summary_stats(actual, pred_probs, num_records,
                                                                                threshold=threshold)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), None


def user_interest_model(records, threshold=.5, start_event=0):
    num_records = float(len(records))

    # for every event
    actual, pred_probs = predict_and_model_cbf(records, is_user_profile=True, start_event=start_event)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, _ = get_summary_stats(actual, pred_probs, num_records,
                                                                                threshold=threshold)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), None


def predict_and_model_ccf(records, transition_mapping, start_event=0):
    actual = []  # as the draw probability cant be zero
    predicted = []
    prev_lect = None

    # start_event = 1
    for idx, event in enumerate(records):
        _, slug, vid_id, part_id, event_time, _, _, label = decode_vector(event)

        current_lect = (slug, vid_id, part_id)

        # if idx == 0:
        #     prev_lect = current_lect
        #     continue

        inter = transition_mapping.value.get((prev_lect, current_lect), 0.)
        union = (transition_mapping.value.get(prev_lect, 0.) + transition_mapping.value.get(current_lect, 0.) - inter)

        if union == 0.:
            sim = 0.
        else:
            sim = inter / union

        assert 0. <= sim <= 1.

        prev_lect = current_lect

        if idx >= start_event:
            actual.append(int(label))
            predicted.append(sim)

    return actual, predicted


def content_collaborative_filtering_model(records, transition_mapping, threshold=0.5, start_event=0):
    num_records = float(len(records))

    # for every event
    actual, pred_probs = predict_and_model_ccf(records, transition_mapping, start_event)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, _ = get_summary_stats(actual, pred_probs, num_records,
                                                                                threshold=threshold)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), None


def predict_and_model_jaccard(records, start_event=0):
    actual = []  # as the draw probability cant be zero
    predicted = []
    topic_dict_p = None

    # start_event = 1

    for idx, event in enumerate(records):
        _, _, _, _, _, _, topic_vec, label = decode_vector(event)
        topic_dict_t = get_topic_dict(topic_vec)

        if idx == 0:
            topic_dict_p = topic_dict_t
            continue

        topic_set_t = set(topic_dict_t.keys())
        topic_set_p = set(topic_dict_p.keys())

        sim = len(topic_set_t.intersection(topic_set_p)) / len(topic_set_t.union(topic_set_p))

        topic_dict_p = topic_dict_t

        if idx >= start_event:
            actual.append(int(label))
            predicted.append(sim)

    return actual, predicted


def jaccard_based_filtering_model(records, threshold=.5, start_event=0):
    num_records = float(len(records))

    # for every event
    actual, pred_probs = predict_and_model_jaccard(records, start_event)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, _ = get_summary_stats(actual, pred_probs, num_records,
                                                                                threshold=threshold)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), None


def user_interest_tfidf_model(records, df_mapping, threshold=.5, start_event=0):
    num_records = float(len(records))

    # for every event
    actual, pred_probs = predict_and_model_cbf(records, is_user_profile=True, df_mapping=df_mapping, start_event=start_event)

    pred_probs = np.array(pred_probs).astype("float64")
    actual = np.array(actual).astype("int")

    accuracy, precision, recall, f1, roc_score, pr_score, _ = get_summary_stats(actual, pred_probs, num_records,
                                                                                threshold=threshold)

    return accuracy, precision, recall, f1, roc_score, pr_score, int(num_records), None
