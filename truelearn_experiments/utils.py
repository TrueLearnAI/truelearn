import gzip
import itertools
import json
import os
from operator import itemgetter
import networkx as nx
import numpy as np
from datetime import datetime as dt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc
from statsmodels.stats.moment_helpers import corr2cov

STATIC_DRAW_PROB = float(0.526475357)  # for full dataset --> 20k users validation


def _get_user_model(user_model):
    _user_model = {}
    # get the topics
    topics = set(user_model["mean"].keys()) | set(user_model["variance"].keys())
    for topic in topics:
        _user_model[topic] = (user_model["mean"][topic], user_model["variance"].get(topic))

    return _user_model


def get_semantic_relatedness_mapping(dir_path):
    # get all the directories
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".gz")]
    mapping = {}
    for f in files:
        with gzip.open(f) as infile:
            for line in infile.readlines():
                tmp_line = json.loads(line)
                a, b = tuple(tmp_line["id_pair"])
                relatedness = float(tmp_line["relatedness"])
                mapping[(a, b)] = relatedness
                mapping[(b, a)] = relatedness

    return mapping


def apply_interest_decay(learner_mean, t, last_t, decay_func):
    t_delta = (dt.utcfromtimestamp(t) - dt.utcfromtimestamp(last_t)).days

    return learner_mean * decay_func(float(t_delta))


def get_summary_stats(actual, pred_probs, num_records, stats=None, user_model=None, threshold=0.5,
                      k_preds=None, i_preds=None, i_stats=None, i_user_model=None, weights=None):
    """

    Args:
        actual:
        predicted:
        stats:
        user_model:

    Returns:

    """
    # validate interest parameters
    if i_preds is not None:
        # know prob must be set and be between 0. and 1.
        assert i_stats is not None and i_user_model is not None

    try:
        roc_score = roc_auc_score(actual, pred_probs)
    except ValueError:
        roc_score = -1.

    pr, rec, thresh = precision_recall_curve(actual, pred_probs, pos_label=1)
    pr_score = auc(rec, pr)

    if all([i == 0 for i in actual]):  # if all the actual values are 0s
        pr_score = -1.

    predicted = (pred_probs >= threshold).astype("int")

    accuracy = accuracy_score(actual, predicted, normalize=True)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    if stats is not None:
        stats["is_hybrid"] = False  # to know if interest data exists or not#

        num_topics = float(stats["num_topics"])

        # get user model
        stats["user_model"] = _get_user_model(user_model)

        # get topic rate
        stats["num_topics_rate"] = num_topics / num_records

        #  get change label rate
        stats["change_label"] = stats.get("change_label", 0.) / num_records

        # in user model
        stats["num_user_topics"] = len(stats["user_model"].keys())

        # get positive rate
        stats["positive"] = len([i for i in actual if i == 1]) / num_records
        stats["predict_positive"] = len([i for i in predicted if i == 1]) / num_records

        stats["actual"] = [float(i) for i in actual.tolist()]
        stats["predicted"] = [float(i) for i in pred_probs.tolist()]

    if i_stats is not None:
        num_topics = float(i_stats["num_topics"])

        # get user model
        i_stats["user_model"] = _get_user_model(i_user_model)

        stats["is_hybrid"] = True

        # get topic rate
        stats["i_num_topics_rate"] = num_topics / num_records

        #  get change label rate
        # stats["i_change_label"] = stats.get("change_label", 0.) / num_records

        # in user model
        stats["i_num_user_topics"] = len(i_stats["user_model"].keys())

        stats["k_preds"] = [float(k) for k in k_preds.tolist()]
        stats["i_preds"] = [float(i) for i in i_preds.tolist()]
        stats["weights"] = weights.tolist()  # for meta-learners

        # record topic experience of the model
        stats["i_topics"] = i_stats["topics"]
        stats["i_num_updates"] = i_stats["num_updates"]

    return accuracy, precision, recall, f1, roc_score, pr_score, stats


def build_graph(nodes, edges):
    import networkx as nx

    # build graph
    g = nx.Graph()
    g.add_nodes_from(nodes)

    g.add_weighted_edges_from(edges)

    return g


def _build_skill_graph(user_model, mapping):
    # get the combinations
    combos = itertools.combinations(user_model["mean"].keys(), 2)
    nodes = user_model["mean"].keys()

    edges = []
    for src, dst in combos:
        rel = mapping.get((src, dst))
        if rel is not None:
            edges.append((src, dst, rel))

    g = build_graph(nodes, edges)

    return g


def convert_to_records(record, top_n=10, has_part_id=False):
    """convert csv of learner events to event records

    Args:
        record (tuple: vals) : (slug, vid_id, part, time, timeframe, session, --topics--, label)
        top_n (int): number of top ranking topics required.

    Returns:
        ({str: val}): record of different attributes of the thingy.
    """
    record = list(record)

    tmp_rec = {
        "slug": str(record[0]),
        "vid_id": int(float(record[1])),
        "label": int(float(record[-1]))
    }

    topic_cutoff = top_n * 2

    if has_part_id:
        topic_vector = record[6:-1][:topic_cutoff]  # topic vector starts at index 6
        tmp_rec["part"] = int(float(record[2]))
        tmp_rec["time"] = float(record[3])
        tmp_rec["timeframe"] = int(float(record[4]))
        tmp_rec["session"] = str(record[5])
        tmp_rec["topics"] = [float(i) for i in topic_vector]
    else:
        topic_vector = record[5:-1][:topic_cutoff]
        tmp_rec["part"] = 1
        tmp_rec["time"] = float(record[2])
        tmp_rec["timeframe"] = int(float(record[3]))
        tmp_rec["session"] = str(record[4])
        tmp_rec["topics"] = [float(i) for i in topic_vector]

    return tmp_rec


def decode_vector(vect, n_topics=10):
    """

    Args:
        vect:
         slug, vid_id, part_id, time, timeframe, --- topic vector --, label

    Returns:

    """
    user_id = vect[0]
    slug = vect[1]
    vid_id = vect[2]
    part_id = vect[3]
    event_time = vect[4]
    event_timeframe = vect[5]

    topic_vec = vect[6:-1][:n_topics * 2]

    label = vect[-1]

    return user_id, slug, vid_id, part_id, event_time, event_timeframe, topic_vec, label


def get_topic_dict(topics, type="cosine"):
    """

    Args:
        topics [float]: a list of numbers where the even indices are the topic ids and odd indices are coverage values
        type (str): type of repr, can be cosine, norm or binary
    Returns:
        {int: float}: a dict with topic id: topic coverage
    """
    num_topics = int(len(topics) / 2)
    topic_dict = {}
    covs = []
    for i in range(num_topics):
        topic_id_idx = i * 2
        topic_cov_idx = topic_id_idx + 1

        topic_id = int(topics[topic_id_idx])
        topic_cov = float(topics[topic_cov_idx])

        topic_dict[topic_id] = topic_cov
        covs.append(topic_cov)

    if type == "cosine":
        return topic_dict
    elif type == "binary":
        return {topic: True for topic in topic_dict}
    else:  # norm transformation
        norm = float(sum(covs))
        return {topic: cov / norm for topic, cov in topic_dict.items()}


def _get_topic_vector(topics, type):
    if type == "cosine":
        return topics
    else:
        n_topics = int(len(topics) / 2)
        new_topics = []

        if type == "binary":
            for i in range(n_topics):
                topic_idx = i * 2
                new_topics += [topics[topic_idx], True]
        elif type == "norm":
            sum_cosine = sum([topics[(i * 2) + 1] for i in range(n_topics)])
            for i in range(n_topics):
                topic_idx = i * 2
                cosine_idx = topic_idx + 1

                new_topics += [topics[topic_idx], topics[cosine_idx] / sum_cosine]

        return new_topics


def vectorise_data(events, vector_type):
    """

    Args:
        events [{key:val}]: dict of the data in the learning event
        vector_type (str): representation of the features (can be binary, cosines or percentage)

    Returns:
        [[user_id, slug, vid_id, part_id, time, timeframe, ... topic_vector ..., label]]: feature vector
    """
    events.sort(key=itemgetter("time", "timeframe", "part"))

    return [[l["session"], l["slug"], l["vid_id"], l["part"], l["time"], l["timeframe"]] +
            _get_topic_vector(l["topics"], vector_type) + [l["label"]] for l in events]


def get_videowise_data(events, is_video=False, is_trueskill=False):
    """Takes a list of events and fragments them into a list of list where every first order list is the collection of
    events for that timeframe.

    Args:
        events:

    Returns:

    """
    if is_trueskill:
        timeframe_idx = "timeframe"
    else:
        timeframe_idx = 4

    video_events = []

    current_timeframe = None
    current_parts = []

    for event in events:

        timeframe = event[timeframe_idx]

        # if first event
        if current_timeframe is None:
            current_timeframe = timeframe

        # if next video
        if timeframe != current_timeframe:
            video_events.append(current_parts)
            current_parts = []
            current_timeframe = timeframe

        current_parts.append(event)

    # last video
    video_events.append(current_parts)

    return video_events


def compute_pr_based_skills(user_model, related_skills, mapping):
    # build the graph
    skill_graph = _build_skill_graph(user_model, mapping)

    # run PageRank
    pr = nx.pagerank(skill_graph, alpha=1., max_iter=100)

    # normalise with pr value
    related_skills = {topic: rel * pr.get(topic, 0.) for topic, rel in related_skills.items()}

    return related_skills


def compute_gaussian_based_skills(user_model, current_topic, related_skills, mapping):
    return related_skills


def get_related_skill_set(user_model, semantic_mapping, topic, top_k_sr_topics, sr_func="raw"):
    related_skills = {}
    for skill in user_model["mean"].keys():
        tmp_rel = semantic_mapping.get((topic, skill), 0.0)
        if tmp_rel > 0.:
            related_skills[skill] = tmp_rel

    if sr_func == "pr":
        try:
            related_skills = compute_pr_based_skills(user_model, related_skills, semantic_mapping)
        except Exception:
            print()

    if top_k_sr_topics == -1:
        return related_skills

    sorted_related_skills = sorted([item for item in related_skills.items()], key=lambda item: -item[1])

    return {topic: rel for (topic, rel) in sorted_related_skills[:top_k_sr_topics]}


def _get_corr_dict(related_skills, semantic_mapping, skill_i, user_model):
    corr_mapping = {}
    std_mapping = {}

    for skill_j, rel in related_skills.items():
        corr_mapping[(skill_j, skill_i)] = corr_mapping[(skill_i, skill_j)] = rel
        std_mapping[skill_j] = np.sqrt(user_model["variance"][skill_j])
        for skill_z in related_skills:
            if skill_j == skill_z:
                corr_mapping[(skill_j, skill_z)] = 1.
            else:
                corr_mapping[(skill_j, skill_z)] = corr_mapping[(skill_z, skill_j)] = semantic_mapping.get(
                    (skill_j, skill_z), 0.)

    return corr_mapping, std_mapping


def _get_corr_arrays(corr_mapping, std_mapping):
    idx_mapping = {skill: idx for idx, skill in enumerate(sorted(std_mapping.keys()))}
    num_topics = len(idx_mapping)
    corr_array = np.zeros(shape=([num_topics, num_topics]))
    std_array = np.zeros(num_topics)

    for j in idx_mapping:
        std_array[idx_mapping[j]] = std_mapping[j]
        for z in idx_mapping:
            corr_array[idx_mapping[j], idx_mapping[z]] = corr_mapping[j, z]

    return corr_array, std_array, idx_mapping


def _get_cov_mapping(cov_array, idx_mapping):
    topic_mapping = {idx: topic for topic, idx in idx_mapping.items()}
    cov_mapping = {}

    for j in topic_mapping:
        for z in topic_mapping:
            # cov_mapping[(topic_mapping[j], topic_mapping[z])] = cov_mapping[(topic_mapping[z], topic_mapping[j])] = cov_array[j, z]
            cov_mapping[(topic_mapping[j], topic_mapping[z])] = cov_array[j, z]

    return cov_mapping


def get_covariance_mapping(related_skills, semantic_mapping, skill_i, user_model):
    # make the data structures
    corr_mapping, std_mapping = _get_corr_dict(related_skills, semantic_mapping, skill_i, user_model)
    corr_array, std_array, idx_mapping = _get_corr_arrays(corr_mapping, std_mapping)

    # get covariance matrix
    cov_array = corr2cov(corr_array, std_array)

    # get covariance mapping
    cov_mapping = _get_cov_mapping(cov_array, idx_mapping)

    return cov_mapping


def compute_gaussian_mean_and_variance(related_skills, prop, user_model, skill_i, semantic_mapping):
    # get covariance mapping
    cov_mapping = get_covariance_mapping(related_skills, semantic_mapping, skill_i, user_model)

    mean = 0.
    var = 0.

    for skill_j, rel_j in related_skills.items():
        mean += prop * rel_j * user_model["mean"][skill_j]
        for skill_z in related_skills:
            if skill_j == skill_z:
                rel_z = 1.
            else:
                rel_z = semantic_mapping[(skill_i, skill_z)]

            var += (prop ** 2) * (rel_j ** 2 * cov_mapping[(skill_j, skill_j)] +
                                  rel_z ** 2 * cov_mapping[(skill_z, skill_z)] +
                                  2 * rel_j * rel_z * cov_mapping[(skill_j, skill_z)])

    return mean, var


def compute_mean_and_variance(related_skills, prop, user_model):
    mean = 0.
    var = 0.

    for skill, rel in related_skills.items():
        mean += prop * rel * user_model["mean"][skill]
        var += (prop * rel) ** 2 * user_model["variance"][skill]

    # var = def_var

    if var == float('Inf'):
        print()

    return mean, var


def get_semantic_skill_inference(user_model, semantic_mapping, topic, init_skill, def_var, event_time, agg_func,
                                 top_k_sr_topics,
                                 sr_func="raw"):
    # find the semantically related topics to current topic
    related_skills = get_related_skill_set(user_model, semantic_mapping, topic, top_k_sr_topics, sr_func=sr_func)

    if len(related_skills) == 0:
        return init_skill, np.sqrt(def_var)

    else:  # if there are related topics
        prop = 1.

        if agg_func == "max":  # if considering only max relatedness
            tmp_skill = sorted(related_skills.items(), key=lambda l: -l[1])[0]
            related_skills = {tmp_skill[0]: tmp_skill[1]}

        if agg_func == "or":  # averaging out using `or` operator
            prop = 1 / len(related_skills)

        if sr_func == "gauss":
            mean, var = compute_gaussian_mean_and_variance(related_skills, prop, user_model, topic, semantic_mapping)
        else:
            mean, var = compute_mean_and_variance(related_skills, prop, user_model)

        return mean, np.sqrt(var)


def _erfc(x):
    """Complementary error function (via `http://bit.ly/zOLqbc`_)"""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
            0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277
    )))
    )))
    )))
    return 2. - r if x < 0 else r


def erfc(x):
    """Complementary error function (via `http://bit.ly/zOLqbc`_)"""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
            0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277
    )))
    )))
    )))
    return 2. - r if x < 0 else r


def _cdf(x, mu=0, sigma=1):
    """Cumulative distribution function"""
    return 0.5 * erfc(-(x - mu) / (sigma * np.sqrt(2)))


def team_sum_quality(mean_skill_user, var_skill_user, mean_skill_content, var_skill_content, beta):
    """Algorithm to compute the quality using the difference of means in a cumulative manner

    Args:
        mean_skill_user ([float): list of means of skills of the learner
        var_skill_user ([float]): list of variances of skills of the learner
        mean_skill_content ([float]): list of means of topics of the content
        var_skill_content ([float]): list of variances of topics of the content
        beta (float): beta value (should be the standard deviation, not the variance)

    Returns:
        (float): probability of mean difference
    """

    difference = np.sum(mean_skill_user) - np.sum(mean_skill_content)
    std = np.sqrt(np.sum(np.sqrt(var_skill_user)) + np.sum(np.sqrt(var_skill_content)) + beta)
    return float(_cdf(difference, 0, std))
